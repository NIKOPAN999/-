"""Phase-one experiments. Development data only; test scoring is intentionally absent."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline"))

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader

import run as legacy
from calibration import apply_calibration, fit_calibration
from data import AlignedDataset, Statistics, block_mask, fit_statistics, random_block_mask
from settings import CONFIG
from ulgm import UnimodalLabelBank
from dataset_io import load_dataset, sha256, video_id
from pooling_model import Phase1Model


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {path}")
    return path


def verify_sources():
    record = json.loads((ROOT / "baseline_manifest.json").read_text(encoding="utf-8"))
    for name, expected in record.items():
        if sha256(ROOT / "baseline" / name) != expected:
            raise ValueError(f"Frozen baseline has changed: {name}")
    return sha256(ROOT / "baseline_manifest.json")


def source_fingerprint():
    names = [ROOT / n for n in ("phase1.py", "pooling_model.py", "dataset_io.py")]
    return {p.name: sha256(p) for p in names}


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None,
            "disk_free_gib": shutil.disk_usage(ROOT).free / 2**30}


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_for(args, schema):
    return {**CONFIG, **schema, "bert_name": str(Path(args.bert).resolve()),
            "pooling": args.pooling, "seed": args.seed,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "mask_probability": args.mask_probability,
            "text_only": getattr(args, "text_only", False)}


def predictions(model, dataset, dev, batch_size, condition=None):
    model.eval()
    labels = {"cls": [], "reg": []}
    scores, logits, removed = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for clean in loader:
            if condition is None:
                batch = clean
            elif condition[0] == "block":
                batch = block_mask(clean, *condition[1])
            else:
                batch = {k: v.clone() if torch.is_tensor(v) else v for k,v in clean.items()}
                for mode in condition[1]:
                    body = batch["content"]
                    if mode == 0:
                        batch["ids"][body] = 103
                    else:
                        batch[("audio", "vision")[mode-1]][body] = 0
                    batch["observed"][:, :, mode] = False
            before = clean["observed"].sum(dim=1).numpy()
            after = batch["observed"].sum(dim=1).numpy()
            removed.append((before-after) / np.maximum(before,1))
            out = model(legacy.to_device(batch, dev))
            logits.append(out["logits"].cpu().numpy())
            scores.append(out["reg"].cpu().numpy())
            labels["cls"].extend(clean["cls"].tolist())
            labels["reg"].extend(clean["reg"].tolist())
    return labels, np.concatenate(logits), np.concatenate(scores), np.concatenate(removed)


def selection_cases(model, dataset, dev, batch):
    result = {}
    for name, condition in legacy.scenarios():
        if name in legacy.SELECTION_VIEWS:
            y,l,s,_ = predictions(model, dataset, dev, batch,
                                  ("block",condition) if condition else None)
            result[name] = y,l,s
    return result


def load_model(path, dev, bert=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("phase1_format") != "pooling_v1":
        del state
        if bert:
            os.environ["SELFMM_BERT_BASE"] = str(Path(bert).resolve())
        return legacy.load_checkpoint(path, dev)
    config = state["config"]
    model = Phase1Model(config, state["bert_config"])
    model.load_state_dict(state["state"], strict=True)
    return model.to(dev), Statistics.from_state_dict(state["statistics"]), state


def preflight(args):
    baseline_hash = verify_sources()
    _, audit = load_dataset(args.data)
    record = {"environment": environment(), "data": audit,
              "frozen_baseline_manifest_sha256": baseline_hash,
              "experiment_sources": source_fingerprint()}
    if args.bert:
        cfg = transformers.AutoConfig.from_pretrained(args.bert, local_files_only=True)
        record["bert_config"] = {k:getattr(cfg,k) for k in ("hidden_size","num_hidden_layers","vocab_size")}
        record["bert_files"] = {p.name:sha256(p) for p in Path(args.bert).glob("*")
                                if p.is_file() and p.suffix in (".json",".bin",".safetensors")}
    write_json(args.output, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


def reproduce(args):
    verify_sources()
    raw, audit = load_dataset(args.data)
    if audit["schema"] != {"length":50,"audio_dim":74,"vision_dim":35}:
        raise ValueError("Contest checkpoint reproduction requires the original 50/74/35 contest dataset")
    out = new_output(args.output)
    model, stats, _ = load_model(ROOT / "baseline/student_compact.pt", device(), args.bert)
    calibration = legacy.read_calibration(ROOT / "baseline/student_compact_calibration.json",
                                         ROOT / "baseline/student_compact.pt")
    ds = AlignedDataset(raw["valid"], stats)
    rows = []
    for name, condition in legacy.scenarios():
        y,l,s,_ = predictions(model, ds, device(), args.eval_batch_size,
                              ("block",condition) if condition else None)
        c,v = apply_calibration(l,s,calibration)
        rows.append({"scenario":name, **legacy.calculate_metrics(y,c,v)})
    with (ROOT / "baseline/compact_valid_calibrated.csv").open(encoding="utf-8-sig") as f:
        expected = {r["scenario"]:r for r in csv.DictReader(f)}
    differences = [{"scenario":r["scenario"], "metric":m,
                    "actual":r[m], "expected":float(expected[r["scenario"]][m]),
                    "absolute_error":abs(r[m]-float(expected[r["scenario"]][m]))}
                   for r in rows for m in legacy.METRICS]
    maximum = max(r["absolute_error"] for r in differences)
    legacy.write_csv(out / "valid_reproduced.csv", rows, ("scenario",*legacy.METRICS))
    record = {"status":"passed" if maximum<=args.tolerance else "mismatch",
              "max_metric_absolute_error":maximum,"tolerance":args.tolerance,
              "differences":differences,"data_sha256":audit["data_sha256"],
              "environment":environment(),"test_evaluated":False}
    write_json(out / "reproduction.json", record)
    print(json.dumps({k:v for k,v in record.items() if k!="differences"},ensure_ascii=False,indent=2))
    if maximum > args.tolerance:
        raise RuntimeError("Baseline mismatch: inspect reproduction.json before starting new training")


def smoke(args):
    verify_sources()
    raw, audit = load_dataset(args.data)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke benchmark requires CUDA")
    out = new_output(args.output)
    stats = fit_statistics(raw["train"])
    ds = AlignedDataset(raw["train"],stats)
    batch = next(iter(DataLoader(ds,batch_size=args.batch_size)))
    results = []
    for pooling in ("last","mean","attention"):
        args.pooling = pooling
        legacy.seed_all(args.seed)
        cfg = config_for(args,audit["schema"])
        model = Phase1Model(cfg).cuda()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            model.train()
            moved = legacy.to_device(batch,device())
            pred = model(moved)
            loss = F.l1_loss(pred["reg"],moved["reg"]) + .8*F.cross_entropy(pred["logits"],moved["cls"])
            loss = loss + .4*sum(F.l1_loss(v,moved["reg"]) for v in pred["aux"].values())/3
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite smoke loss")
        results.append({"pooling":pooling,"batch_size":args.batch_size,"steps":3,
                        "seconds":seconds,"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                        "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                        "parameters":sum(p.numel() for p in model.parameters()),
                        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)})
        del model,optimizer,pred,loss,moved
        torch.cuda.empty_cache()
    write_json(out / "smoke.json", {"results":results,"environment":environment(),
               "scope":"3 optimizer steps per pooling; not a full-epoch timing or scientific result"})
    print(json.dumps(results,indent=2))


def train(args):
    verify_sources()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    raw,audit = load_dataset(args.data)
    out = new_output(args.output)
    cfg = config_for(args,audit["schema"])
    legacy.seed_all(cfg["seed"])
    stats = fit_statistics(raw["train"])
    train_ds,valid_ds = AlignedDataset(raw["train"],stats),AlignedDataset(raw["valid"],stats)
    # Preserve the original DataLoader RNG behavior for baseline continuity.
    loader = DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=True)
    model = Phase1Model(cfg).cuda()
    optimizer = torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("bert.")],"lr":cfg["bert_lr"]},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("bert.")],"lr":cfg["lr"]},
    ],weight_decay=cfg["weight_decay"])
    bank = UnimodalLabelBank(torch.from_numpy(train_ds.reg),cfg["hidden"])
    mask_generator = torch.Generator().manual_seed(cfg["seed"]+17)
    write_json(out/"config.json",cfg)
    write_json(out/"provenance.json",{"data":audit,"environment":environment(),
               "source_hashes":source_fingerprint(),"test_evaluated":False,
               "selection":"original six validation views and original uncalibrated composite score",
               "comparison":"full-precision checkpoints; compression/calibration effects reported separately"})
    best,stale = -math.inf,0
    checkpoint = out/"best.pt"
    start = time.perf_counter()
    for epoch in range(1,cfg["epochs"]+1):
        model.train(); bank.begin_epoch()
        losses=[]; epoch_start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for clean in loader:
            damaged=random_block_mask(clean,mask_generator,cfg["mask_probability"],cfg["text_bias"],
                                      cfg["max_mask_ratio"],cfg["triple_probability"])
            batch=legacy.to_device(damaged,device())
            pred=model(batch)
            truth_reg,truth_cls=clean["reg"].cuda(),clean["cls"].cuda()
            pseudo=bank.targets(clean["index"],device())
            ce=F.cross_entropy(pred["logits"],truth_cls,reduction="none")
            weight=1+(cfg["neutral_class_weight"]-1)*(truth_cls==1).float()
            ce=(ce*weight).sum()/weight.sum()
            auxiliary=[]
            for j,mode in enumerate(legacy.MODES):
                available=(pred["availability"][:,j]>0).float()
                difference=torch.tanh((pseudo[mode]-truth_reg).abs())
                auxiliary.append(((pred["aux"][mode]-pseudo[mode]).abs()*difference*available).sum()/available.sum().clamp_min(1))
            loss=cfg["regression_weight"]*F.l1_loss(pred["reg"],truth_reg)+cfg["classification_weight"]*ce+cfg["auxiliary_weight"]*sum(auxiliary)/3
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            bank.record(clean["index"],pred["features"])
            losses.append(float(loss.detach()))
        update=bank.update(epoch)
        cases=selection_cases(model,valid_ds,device(),args.eval_batch_size)
        rows=legacy.evaluate_cases(cases)
        score=legacy.selection_score(rows)
        if not np.isfinite(np.mean(losses)) or not np.isfinite(score):
            raise RuntimeError("Nonfinite training/validation result")
        record={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_score":score,
                "validation":rows,"pseudo_update":update,"seconds":time.perf_counter()-epoch_start,
                "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
        with (out/"epochs.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        print(f"epoch={epoch} pool={cfg['pooling']} seed={cfg['seed']} loss={record['train_loss']:.5f} score={score:.5f} seconds={record['seconds']:.1f}",flush=True)
        if score>best+1e-5:
            best,stale=score,0
            torch.save({"phase1_format":"pooling_v1","state":model.state_dict(),"config":cfg,
                        "statistics":stats.state_dict(),"bert_config":model.bert.config.to_dict(),
                        "epoch":epoch,"selection_score":score},checkpoint)
        else:
            stale+=1
            if stale>=cfg["patience"]:
                break
    del model,optimizer
    torch.cuda.empty_cache()
    chosen,_,selected=load_model(checkpoint,device())
    cases=selection_cases(chosen,valid_ds,device(),args.eval_batch_size)
    fitted=fit_calibration(cases,"accuracy",.01)
    fitted["checkpoint_sha256"]=sha256(checkpoint)
    write_json(out/"calibration.json",fitted)
    write_json(out/"training_complete.json",{"selected_epoch":selected["epoch"],
               "checkpoint_sha256":sha256(checkpoint),"elapsed_seconds":time.perf_counter()-start,
               "test_evaluated":False})
    print(f"Training complete: {checkpoint}; run diagnose to export all validation scenarios.",flush=True)


def diagnose(args):
    verify_sources()
    raw,audit=load_dataset(args.data)
    out=new_output(args.output)
    model,stats,state=load_model(args.checkpoint,device(),args.bert)
    if args.calibration:
        obj=json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        if obj.get("checkpoint_sha256")!=sha256(args.checkpoint):
            raise ValueError("Calibration/checkpoint SHA-256 mismatch")
        calibration=obj["parameters"]
    else:
        calibration=None
    ds=AlignedDataset(raw["valid"],stats)
    views=[(n,("block",c) if c else None) for n,c in legacy.scenarios()]
    views += [("drop_text",("drop",(0,))),("drop_audio",("drop",(1,))),
              ("drop_vision",("drop",(2,))),("drop_audio_vision",("drop",(1,2)))]
    summary=[]
    clean_logits=clean_scores=None
    fields=("sample_id","video_id","scenario","true_class","true_intensity","raw_score",
            "raw_class","class","intensity","logit_negative","logit_neutral","logit_positive",
            "raw_score_absolute_change","class_changed","removed_observed_text_fraction",
            "removed_observed_audio_fraction","removed_observed_vision_fraction")
    with (out/"samples.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for name,condition in views:
            y,l,s,removed=predictions(model,ds,device(),args.eval_batch_size,condition)
            raw_class,raw_projected=apply_calibration(l,s)
            classes,values=apply_calibration(l,s,calibration)
            if name=="clean":
                clean_logits,clean_scores=l.copy(),s.copy()
            metrics=legacy.calculate_metrics(y,classes,values)
            raw_metrics=legacy.calculate_metrics(y,raw_class,raw_projected)
            summary.append({"scenario":name,**metrics,
                            "raw_accuracy":raw_metrics["accuracy"],"raw_macro_f1":raw_metrics["macro_f1"],
                            "raw_mae":float(np.mean(np.abs(np.asarray(y["reg"])-s))),
                            "raw_pearson":float(np.corrcoef(y["reg"],s)[0,1]) if np.std(s)>0 else None,
                            "mean_absolute_raw_score_change":float(np.mean(np.abs(s-clean_scores))),
                            "raw_class_flip_fraction":float(np.mean(l.argmax(-1)!=clean_logits.argmax(-1))),
                            **{f"mean_removed_observed_{m}_fraction":float(removed[:,j].mean()) for j,m in enumerate(legacy.MODES)}})
            for i,sid in enumerate(ds.sample_ids):
                writer.writerow(dict(sample_id=sid,video_id=video_id(sid) or "",scenario=name,
                    true_class=y["cls"][i],true_intensity=y["reg"][i],raw_score=float(s[i]),
                    raw_class=int(raw_class[i]),**{"class":int(classes[i])},intensity=float(values[i]),
                    logit_negative=float(l[i,0]),logit_neutral=float(l[i,1]),logit_positive=float(l[i,2]),
                    raw_score_absolute_change=float(abs(s[i]-clean_scores[i])),
                    class_changed=int(l[i].argmax()!=clean_logits[i].argmax()),
                    removed_observed_text_fraction=float(removed[i,0]),
                    removed_observed_audio_fraction=float(removed[i,1]),removed_observed_vision_fraction=float(removed[i,2])))
            print(f"{name}: accuracy={metrics['accuracy']:.4f} raw_mae={summary[-1]['raw_mae']:.4f}",flush=True)
    legacy.write_csv(out/"metrics.csv",summary,tuple(summary[0]))
    write_json(out/"diagnostic_complete.json",{"config":state["config"],"checkpoint_sha256":sha256(args.checkpoint),
               "data_sha256":audit["data_sha256"],"environment":environment(),"test_evaluated":False,
               "calibration":str(args.calibration) if args.calibration else None,
               "scope":"Validation interventions; sensitivity alone is not evidence of useful complementary information."})


def summarize(args):
    results=[]
    for path in sorted(Path(args.runs).glob("*/diagnostics/diagnostic_complete.json")):
        meta=json.loads(path.read_text(encoding="utf-8")); cfg=meta["config"]
        with path.with_name("metrics.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                results.append({"run":path.parent.parent.name,"pooling":cfg.get("pooling","legacy"),
                                "seed":cfg["seed"],"text_only":cfg.get("text_only",False),
                                "mask_probability":cfg["mask_probability"],"data_sha256":meta["data_sha256"],**r})
    if not results:
        raise ValueError("No completed diagnostics found")
    legacy.write_csv(args.output,results,tuple(results[0]))
    print(f"Exported {len(results)} rows to {args.output}; no automatic hypothesis acceptance.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    for name in ("preflight","reproduce","smoke","train","diagnose"):
        q=subs.add_parser(name)
        q.add_argument("--data",type=Path,required=True)
        q.add_argument("--bert",required=name in ("reproduce","smoke","train"))
        q.add_argument("--output",type=Path,required=True)
        if name in ("reproduce","train","diagnose"):
            q.add_argument("--eval-batch-size",type=int,default=8)
        if name in ("smoke","train"):
            q.add_argument("--batch-size",type=int,default=8)
            q.add_argument("--seed",type=int,default=42)
            q.add_argument("--pooling",choices=("last","mean","attention"),default="last")
            q.add_argument("--epochs",type=int,default=20)
            q.add_argument("--mask-probability",type=float,default=.55)
            q.add_argument("--text-only",action="store_true")
        if name=="reproduce":
            q.add_argument("--tolerance",type=float,default=.001)
        if name=="diagnose":
            q.add_argument("--checkpoint",type=Path,required=True)
            q.add_argument("--calibration",type=Path)
    q=subs.add_parser("summarize")
    q.add_argument("--runs",type=Path,required=True)
    q.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if hasattr(args,"mask_probability") and not 0<=args.mask_probability<=1:
        p.error("mask probability must be in [0,1]")
    globals()[args.command](args)


if __name__=="__main__":
    main()
