"""Restore exact full-precision checkpoints using the verified frozen BERT base."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'baseline'))
import torch
import run as legacy
from pooling_model import Phase1Model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delta',type=Path,required=True)
    p.add_argument('--bert',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    obj=torch.load(args.delta,map_location='cpu',weights_only=False)
    if obj.get('backup_format')!='phase1_delta_fp32_v1':raise ValueError('Unknown backup format')
    cfg=dict(obj['config'],bert_name=str(args.bert.resolve()))
    model=Phase1Model(cfg)
    if legacy.base_fingerprint(model.state_dict(),cfg['freeze_bert_layers'])!=obj['base_fingerprint']:
        raise ValueError('Frozen BERT fingerprint mismatch')
    missing,unexpected=model.load_state_dict(obj['state'],strict=False)
    if unexpected or any(not legacy.frozen_name(k,cfg['freeze_bert_layers']) for k in missing):
        raise ValueError('Backup keys mismatch')
    obj['state']=model.state_dict()
    # Keep the original config so an existing calibration SHA does not silently
    # get rebound. Restored archive files have a new file hash; recalibration
    # metadata must be explicitly rebound after prediction-equivalence checks.
    obj.pop('backup_format',None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,args.output)
    print('Restored exact tensor values. File SHA differs; verify predictions before binding calibration metadata.')


if __name__=='__main__':main()
