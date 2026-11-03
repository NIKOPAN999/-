"""Behavioral checks for controlled pooling, masks and held-out-data boundaries."""
import copy
import gc
import pickle
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"baseline"))
import numpy as np
from dataset_io import load_dataset,video_id

try:
    import torch
    from transformers import BertConfig
    from model import SelfMMContest
    from pooling_model import Phase1Model,ObservedPoolingEncoder
    TORCH=True
except ImportError:
    TORCH=False


def fixture():
    result={}
    for split in ("train","valid","test"):
        t=np.zeros((3,3,6),dtype=np.int64)
        t[:,0]=[101,7,8,9,102,0]; t[:,1]=[1,1,1,1,1,0]
        a=np.ones((3,6,74),dtype=np.float32); v=np.ones((3,6,35),dtype=np.float32)
        result[split]={"text_bert":t,"audio":a,"vision":v,
                       "id":[f"{split}_video{i}$_$0" for i in range(3)]}
        if split!="test":
            result[split].update(regression_labels=np.array([-1.,0.,1.]),classification_labels=np.array([0,1,2]))
    return result


class DatasetTests(unittest.TestCase):
    def load(self,obj):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"data.pkl"
            with path.open("wb") as f:pickle.dump(obj,f)
            return load_dataset(path)

    def test_test_labels_are_not_required_or_reported(self):
        data,audit=self.load(fixture())
        self.assertNotIn("test",data)
        self.assertNotIn("class_counts",audit["splits"]["test"])
        self.assertTrue(audit["video_disjointness_verified"])

    def test_original_video_overlap_is_rejected(self):
        obj=fixture();obj["valid"]["id"][0]="train_video0$_$other_clip"
        with self.assertRaisesRegex(ValueError,"video overlap"):self.load(obj)

    def test_attention_holes_are_rejected(self):
        obj=fixture();obj["train"]["text_bert"][0,1]=[1,1,0,1,1,0]
        with self.assertRaisesRegex(ValueError,"contiguous prefix"):self.load(obj)

    def test_unaligned_features_are_rejected(self):
        obj=fixture();obj["train"]["audio"]=obj["train"]["audio"][:,:3]
        with self.assertRaisesRegex(ValueError,"genuinely aligned"):self.load(obj)

    def test_unknown_group_ids_do_not_claim_disjointness(self):
        obj=fixture()
        for split in obj:obj[split]["id"]=[f"{split}_{i}" for i in range(3)]
        _,audit=self.load(obj)
        self.assertFalse(audit["video_disjointness_verified"])
        self.assertEqual(video_id("abc[12]"),"abc")


@unittest.skipUnless(TORCH,"torch/transformers not installed; pooling tests require server environment")
class PoolingTests(unittest.TestCase):
    def configs(self):
        bert=BertConfig(vocab_size=200,hidden_size=16,num_hidden_layers=2,
                        num_attention_heads=2,intermediate_size=32,
                        hidden_dropout_prob=0.,attention_probs_dropout_prob=0.).to_dict()
        cfg=dict(bert_name="unused",hidden=8,dropout=0.,freeze_bert_layers=1,
                 audio_dim=74,vision_dim=35,pooling="last",text_only=False)
        return cfg,bert

    def batch(self):
        t=fixture()["train"]["text_bert"][:2]
        return {"ids":torch.tensor(t[:,0]),"attn":torch.tensor(t[:,1]).bool(),
                "segment":torch.tensor(t[:,2]),"content":torch.tensor([[0,1,1,1,0,0]]*2).bool(),
                "observed":torch.tensor([[[0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0]]]*2).bool(),
                "audio":torch.randn(2,6,74),"vision":torch.randn(2,6,35)}

    def test_last_model_matches_original_bitwise(self):
        cfg,bc=self.configs()
        torch.manual_seed(42);old=SelfMMContest("unused",8,0.,1,bc).eval()
        torch.manual_seed(42);new=Phase1Model(cfg,bc).eval()
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for key,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        batch=self.batch()
        with torch.no_grad():a,b=old(batch),new(batch)
        for key in ("reg","logits"):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_initial_attention_equals_observed_mean(self):
        cfg,bc=self.configs();cfg["pooling"]="mean"
        torch.manual_seed(42);mean=Phase1Model(cfg,bc).eval()
        cfg["pooling"]="attention"
        torch.manual_seed(42);att=Phase1Model(cfg,bc).eval()
        batch=self.batch()
        with torch.no_grad():a,b=mean(batch),att(batch)
        torch.testing.assert_close(a["reg"],b["reg"],rtol=1e-6,atol=1e-6)

    def test_missing_modalities_are_finite_zeros_and_padding_is_ignored(self):
        for pooling in ("last","mean","attention"):
            encoder=ObservedPoolingEncoder(3,4,0.,pooling).eval()
            x=torch.randn(2,6,3)
            observed=torch.tensor([[0,1,1,0,0,0],[0,0,0,0,0,0]]).bool()
            length=torch.tensor([4,3])
            a=encoder(x,observed,length)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a[1],torch.zeros(4),rtol=0,atol=0)
            changed=x.clone();changed[:,4:]=9999
            torch.testing.assert_close(a,encoder(changed,observed,length),rtol=0,atol=0)

    def test_mean_pooling_backpropagates_to_early_observed_inputs(self):
        torch.manual_seed(7)
        enc=ObservedPoolingEncoder(3,4,0.,"mean")
        x=torch.randn(1,7,3,requires_grad=True)
        observed=torch.tensor([[0,1,1,1,1,1,0]]).bool()
        enc(x,observed,torch.tensor([7]))[0,0].backward()
        self.assertGreater(float(x.grad[0,1].abs().sum()),0.)
        torch.testing.assert_close(x.grad[0,0],torch.zeros(3),rtol=0,atol=0)

    def test_exact_delta_backup_can_restore_every_tensor(self):
        import run as legacy
        from transformers import AutoModel
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/'bert'
            bc=BertConfig(vocab_size=200,hidden_size=16,num_hidden_layers=12,
                          num_attention_heads=2,intermediate_size=32)
            # Avoid a Windows safetensors memory map outliving this temp folder.
            AutoModel.from_config(bc).save_pretrained(base,safe_serialization=False)
            cfg=dict(bert_name=str(base),hidden=8,dropout=0.,freeze_bert_layers=8,
                     audio_dim=5,vision_dim=20,pooling='attention',text_only=False)
            model=Phase1Model(cfg)
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:param.add_(.01)
            original={k:v.clone() for k,v in model.state_dict().items()}
            delta={'phase1_format':'pooling_v1','backup_format':'phase1_delta_fp32_v1',
                   'config':cfg,'bert_config':model.bert.config.to_dict(),
                   'state':{k:v for k,v in original.items() if not legacy.frozen_name(k,8)},
                   'base_fingerprint':legacy.base_fingerprint(original,8)}
            torch.save(delta,root/'delta.pt')
            subprocess.run([sys.executable,str(ROOT/'restore_delta.py'),'--delta',str(root/'delta.pt'),
                            '--bert',str(base),'--output',str(root/'restored.pt')],
                           check=True,capture_output=True,text=True)
            restored=torch.load(root/'restored.pt',map_location='cpu',weights_only=False)
            self.assertEqual(set(original),set(restored['state']))
            for k,v in original.items():torch.testing.assert_close(v,restored['state'][k],rtol=0,atol=0)
            del model,original,delta,restored
            gc.collect()


if __name__=="__main__":
    unittest.main(verbosity=2)
