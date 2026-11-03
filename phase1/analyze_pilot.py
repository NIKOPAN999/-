"""Summarize raw held-development-set outcomes without declaring significance."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    records=[]
    for path in sorted(args.runs.glob("*/diagnostics/diagnostic_complete.json")):
        meta=json.loads(path.read_text(encoding="utf-8"))
        with path.with_name("metrics.csv").open(encoding="utf-8-sig") as f:
            rows={r["scenario"]:r for r in csv.DictReader(f)}
        cfg=meta["config"]
        clean=rows["clean"]; dropped=rows["drop_audio_vision"]
        record={"run":path.parent.parent.name,"seed":cfg["seed"],
                "pooling":cfg.get("pooling","legacy"),"text_only":cfg.get("text_only",False),
                "data_sha256":meta["data_sha256"],
                "clean_raw_accuracy":float(clean["raw_accuracy"]),
                "clean_raw_mae":float(clean["raw_mae"]),
                "drop_av_raw_accuracy":float(dropped["raw_accuracy"]),
                "drop_av_raw_mae":float(dropped["raw_mae"]),
                "drop_av_accuracy_change_pp":100*(float(dropped["raw_accuracy"])-float(clean["raw_accuracy"])),
                "drop_av_raw_score_change":float(dropped["mean_absolute_raw_score_change"])}
        for mode in ("audio","vision"):
            for pos in ("front","middle","end"):
                row=rows[f"{mode}_0.5_{pos}"]
                record[f"{mode}_half_{pos}_score_change"]=float(row["mean_absolute_raw_score_change"])
                record[f"{mode}_half_{pos}_class_flip"]=float(row["raw_class_flip_fraction"])
                record[f"{mode}_half_{pos}_observed_removed"]=float(row[f"mean_removed_observed_{mode}_fraction"])
        records.append(record)
    if not records:raise ValueError("No completed diagnostics")
    references={(r["data_sha256"],r["seed"]):r for r in records if r["text_only"]}
    for r in records:
        ref=references.get((r["data_sha256"],r["seed"]))
        r["clean_accuracy_gain_over_text_only_pp"]=100*(r["clean_raw_accuracy"]-ref["clean_raw_accuracy"]) if ref else None
        r["clean_raw_mae_gain_over_text_only"]=ref["clean_raw_mae"]-r["clean_raw_mae"] if ref else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(records,indent=2))
    print("Exploratory validation results. Small drops or large attribution scores alone do not establish useful multimodal robustness.")


if __name__=="__main__":main()
