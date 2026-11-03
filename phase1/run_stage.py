"""Sequential launch with explicit verification/pilot/multiseed stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("verify","pilot","multiseed"),required=True)
    p.add_argument("--dataset",choices=("contest","mosi"),required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--bert",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--eval-batch-size",type=int,default=8)
    args=p.parse_args()
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    shared=["--data",str(args.data.resolve()),"--bert",str(args.bert.resolve())]

    def call(command,*extra):
        cmd=[sys.executable,"-u",str(ROOT/"phase1.py"),command,*shared,*map(str,extra)]
        print("RUN",json.dumps(cmd,ensure_ascii=False),flush=True)
        subprocess.run(cmd,check=True)

    if args.stage=="verify":
        call("preflight","--output",root/"preflight.json")
        if args.dataset=="contest":
            call("reproduce","--output",root/"reproduction","--eval-batch-size",args.eval_batch_size)
            call("diagnose","--checkpoint",ROOT/"baseline/student_compact.pt",
                 "--calibration",ROOT/"baseline/student_compact_calibration.json",
                 "--output",root/"baseline_diagnostics","--eval-batch-size",args.eval_batch_size)
        call("smoke","--output",root/"gpu_smoke","--batch-size",args.batch_size)
        (root/"verification_complete.json").write_text(json.dumps({
            "dataset":args.dataset,"data":str(args.data.resolve()),
            "bert":str(args.bert.resolve()),"batch_size":args.batch_size,
            "eval_batch_size":args.eval_batch_size,
            "data_sha256":json.loads((root/"preflight.json").read_text())["data"]["data_sha256"],
            "test_evaluated":False},indent=2),encoding="utf-8")
        print("Verification finished. Inspect reproduction, data audit and peak memory before pilot.")
        return
    gate=json.loads((root/"verification_complete.json").read_text())
    if gate["dataset"]!=args.dataset or gate["data"]!=str(args.data.resolve()) or gate["bert"]!=str(args.bert.resolve()):
        raise ValueError("Verification belongs to different data/BERT paths")
    if (gate["batch_size"],gate["eval_batch_size"])!=(args.batch_size,args.eval_batch_size):
        raise ValueError("Changed batch sizes require a new verification/output directory")
    from dataset_io import sha256
    if gate["data_sha256"]!=sha256(args.data):
        raise ValueError("Data changed since verification")
    if args.stage=="multiseed":
        for name in ("last","mean","attention","text_only"):
            if not (root/"runs"/f"{name}_s42"/"diagnostics/diagnostic_complete.json").is_file():
                raise ValueError("Complete and inspect the seed-42 pilot before multiseed")
    for seed in ((42,) if args.stage=="pilot" else (43,44)):
        for name in ("last","mean","attention","text_only"):
            run=root/"runs"/f"{name}_s{seed}"
            call("train","--pooling","last" if name=="text_only" else name,
                 "--seed",seed,"--batch-size",args.batch_size,"--eval-batch-size",args.eval_batch_size,
                 "--output",run,*( ["--text-only"] if name=="text_only" else []))
            call("diagnose","--checkpoint",run/"best.pt","--calibration",run/"calibration.json",
                 "--output",run/"diagnostics","--eval-batch-size",args.eval_batch_size)
    subprocess.run([sys.executable,str(ROOT/"phase1.py"),"summarize","--runs",str(root/"runs"),
                    "--output",str(root/"all_validation_metrics.csv")],check=True)


if __name__=="__main__":
    main()
