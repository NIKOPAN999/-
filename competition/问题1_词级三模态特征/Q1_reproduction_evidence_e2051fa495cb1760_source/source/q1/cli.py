from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .core import fingerprint, make_manifest, read_json, sha256, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]


def environment():
    packages = ["numpy", "torch", "torchaudio", "transformers", "whisperx", "opensmile", "openpyxl"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def load_config(args):
    config = read_json(args.config)
    for name in ["data_root", "work_root", "bert_model_path", "align_model_path", "model_provenance",
                 "mfa_dictionary", "mfa_acoustic", "openface_model_dir"]:
        value = getattr(args, name, None) or config[name]
        path = Path(value).expanduser()
        config[name] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    if args.device:
        config["device"] = args.device
    if args.face_reviews:
        config["face_reviews"] = read_json(args.face_reviews)
    return config


def external_identity(config):
    """Hash the actual native executables and external model files used by this run."""
    identity = {}
    for key in ["ffmpeg", "ffprobe", "mfa_bin", "openface_bin"]:
        path = shutil.which(config[key])
        if path is None:
            raise FileNotFoundError(f"Missing executable: {key}={config[key]}")
        identity[key] = dict(path=path, sha256=sha256(path))
    for key in ["mfa_dictionary", "mfa_acoustic"]:
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f"Configure a local downloaded MFA model zip: {key}={path}")
        identity[key] = dict(path=str(path), sha256=sha256(path))
    directory = Path(config["openface_model_dir"])
    files = [p for p in directory.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError("Configure openface_model_dir to the installed OpenFace model directory")
    identity["openface_models"] = {p.relative_to(directory).as_posix(): sha256(p) for p in files}
    au_directory = Path(config["openface_bin"]).parent / "AU_predictors"
    au_files = sorted(p for p in au_directory.rglob("*") if p.is_file())
    if not au_files:
        raise FileNotFoundError("OpenFace AU_predictors directory is missing")
    identity["openface_au_models"] = {p.relative_to(au_directory).as_posix(): sha256(p) for p in au_files}
    identity["mfa_version"] = subprocess.check_output([config["mfa_bin"], "version"], timeout=60,
                                                      stderr=subprocess.STDOUT).decode(errors="replace").strip()
    identity["ffmpeg_version"] = subprocess.check_output([config["ffmpeg"], "-version"], timeout=20).decode(errors="replace").splitlines()[0]
    return identity


def context(config, require_models=False):
    manifest_path = Path(config["work_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    labels = list(Path(config["data_root"]).rglob("label-100.xlsx"))
    if len(labels) != 1 or sha256(labels[0]) != manifest["label_sha256"]:
        raise ValueError("Label source changed: regenerate manifest before continuing")
    provenance = Path(config["model_provenance"])
    if require_models and not provenance.is_file():
        raise FileNotFoundError("Model provenance missing; run explicit download-models stage on server")
    models = read_json(provenance) if provenance.is_file() else None
    if require_models:
        for entry in models["models"].values():
            for rel, digest in entry["sha256"].items():
                path = Path(entry["path"]) / rel
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError(f"Model changed or missing: {path}")
    source = {p.name: sha256(p) for p in (ROOT / "q1").glob("*.py")}
    external = external_identity(config) if require_models else None
    signature = fingerprint(dict(config=config, manifest=manifest, code=source,
                                 models=models, external=external, environment=environment()))[:16]
    run = Path(config["work_root"]) / "runs" / signature
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "run.json", dict(signature=signature, config=config, code=source,
                                      models=models, external=external, environment=environment(),
                                      created_or_resumed=datetime.now(timezone.utc).isoformat()))
    write_json(Path(config["work_root"]) / "latest_run.json", dict(path=str(run)))
    return manifest, run


def sample(manifest, sample_id):
    if sample_id is None:
        raise ValueError("Specify --sample with the exact sample_id from plan or manifest")
    matches = [r for r in manifest["samples"] if r["sample_id"] == sample_id or r["key"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one sample matching {sample_id}")
    return matches[0]


def download_models(config):
    from huggingface_hub import HfApi, snapshot_download
    provenance = dict(models={})
    for key, path_key, repo in [("bert", "bert_model_path", "google-bert/bert-base-uncased"),
                                ("align", "align_model_path", "facebook/wav2vec2-base-960h")]:
        revision = HfApi().model_info(repo, revision=config.get(key + "_revision", "main")).sha
        directory = Path(config[path_key])
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to mix model snapshots in {directory}; use an empty model directory")
        # Prefer safetensors to avoid downloading duplicate legacy PyTorch weights.
        snapshot_download(repo_id=repo, revision=revision, local_dir=directory,
                          allow_patterns=["*.json", "*.txt", "*.safetensors"])
        hashes = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*")
                  if p.is_file() and ".cache" not in p.parts}
        provenance["models"][key] = dict(repo=repo, revision=revision, path=str(directory), sha256=hashes)
    write_json(config["model_provenance"], provenance)
    print("Downloaded model revisions recorded in", config["model_provenance"])


def doctor(config, gpu):
    report = environment()
    report["executables"] = {}
    for key in ["ffmpeg", "ffprobe", "mfa_bin", "openface_bin"]:
        path = shutil.which(config[key])
        report["executables"][key] = path
    report["models_ready"] = all((Path(config[k]) / "config.json").is_file()
                                 for k in ["bert_model_path", "align_model_path"])
    report["mfa_models_ready"] = all(Path(config[k]).is_file() for k in ["mfa_dictionary", "mfa_acoustic"])
    report["openface_models_ready"] = Path(config["openface_model_dir"]).is_dir()
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU check requested but CUDA is unavailable")
        value = (torch.ones(4, device="cuda") * 2).sum().item()
        report["gpu"] = dict(name=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                             capability=torch.cuda.get_device_capability(0), tensor_check=value)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(report["executables"].values()) and all(report[k] for k in
                ["models_ready", "mfa_models_ready", "openface_models_ready"]) else 1


def verify_complete(directory):
    marker = Path(directory) / "complete.json"
    if not marker.is_file():
        return False
    try:
        metadata = read_json(marker)
        required = {"aligned.npz", "native.npz", "metadata.json", "alignment_" + metadata["backend"] + ".json"}
        if set(metadata["files"]) != required:
            return False
        return all((Path(directory) / name).is_file() and sha256(Path(directory) / name) == digest
                   for name, digest in metadata["files"].items())
    except (KeyError, ValueError):
        return False


def extract_all(config, manifest, run, args):
    from .aligners import Aligners
    from .features import Extractors, extract
    from .media import prepare
    decision = read_json(run / "selection.json")
    backend = decision["backend"]
    if decision["run_signature"] != run.name:
        raise ValueError("Aligner selection belongs to a different configuration")
    rows = [sample(manifest, args.sample)] if args.sample else manifest["samples"]
    aligners, extractors = Aligners(config), Extractors(config)
    failures = 0
    for i, row in enumerate(rows, 1):
        directory = run / "samples" / row["key"]
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(rows)}] {row['sample_id']}", flush=True)
        source = Path(config["data_root"]) / row["relative_path"]
        try:
            if sha256(source) != row["sha256"]:
                raise ValueError("Source hash mismatch")
            if verify_complete(directory):
                print("  verified cache", flush=True)
                continue
            media = prepare(config, row, directory)
            alignment = aligners.run(backend, row, directory, media)
            quality, warnings = extract(config, extractors, row, directory, media, alignment)
            files = ["aligned.npz", "native.npz", "metadata.json", f"alignment_{backend}.json"]
            write_json(directory / "complete.json", dict(files={p: sha256(directory / p) for p in files},
                                                         quality=quality, warnings=warnings,
                                                         backend=backend, status="needs_review" if warnings else "ok"))
            # Retain original video + native features; decoded PNGs are regenerable.
            # Delete only our known frame cache after checksums verify, never recursively.
            frames = directory / "frames"
            if verify_complete(directory) and frames.is_dir() and not frames.is_symlink():
                if frames.resolve().parent != directory.resolve():
                    raise ValueError("Frame cache escapes sample directory")
                for frame in frames.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].png"):
                    frame.unlink()
                (frames / "complete.json").unlink(missing_ok=True)
        except Exception as exc:
            failures += 1
            write_json(directory / "failure.json", dict(sample_id=row["sample_id"], error=str(exc),
                                                       traceback=traceback.format_exc()))
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
    summary = []
    for row in manifest["samples"]:
        directory = run / "samples" / row["key"]
        record = dict(sample_id=row["sample_id"], status="not_processed", words=row["words"], duration="",
                      alignment_coverage="", audio_word_coverage="", vision_word_coverage="",
                      text_dim="", audio_dim="", vision_dim="", warnings="")
        if verify_complete(directory):
            done = read_json(directory / "complete.json")
            record.update({k: v for k, v in done["quality"].items() if k in record})
            record.update(status=done["status"], warnings=";".join(done["warnings"]))
        elif (directory / "failure.json").is_file():
            record.update(status="failed", warnings=read_json(directory / "failure.json")["error"])
        summary.append(record)
    write_csv(run / "summary.csv", summary)
    print("Summary:", run / "summary.csv")
    return 1 if failures else 0


def bundle(config, output):
    manifest = make_manifest(config["data_root"])
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    sources = [*ROOT.glob("*.md"), *ROOT.glob("requirements*.txt")]
    for folder in ["q1", "configs", "scripts", "tests"]:
        sources += [p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    with tarfile.open(target, "w:gz") as tar:
        for source in sources:
            tar.add(source, arcname="mosei_q1/" + source.relative_to(ROOT).as_posix())
        for row in manifest["samples"]:
            source = Path(config["data_root"]) / row["relative_path"]
            tar.add(source, arcname="mosei_q1/data/raw100/" + row["relative_path"])
        labels = next(Path(config["data_root"]).rglob("label-100.xlsx"))
        tar.add(labels, arcname="mosei_q1/data/raw100/label-100.xlsx")
    print("Bundle:", target, "bytes:", target.stat().st_size)
    print("Includes code and attachment 1 only; excludes host.md, credentials, models and work outputs.")


def package_outputs(manifest, run, output):
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(target)
    for row in manifest["samples"]:
        directory = run / "samples" / row["key"]
        if not verify_complete(directory):
            raise ValueError(f"Incomplete outputs: {row['sample_id']}")
        if read_json(directory / "complete.json")["warnings"]:
            raise ValueError(f"Unresolved review flags: {row['sample_id']}. Do not hide missing/ambiguous evidence.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in manifest["samples"]:
            directory = run / "samples" / row["key"]
            for name in ["aligned.npz", "native.npz", "alignment_" + read_json(run / "selection.json")["backend"] + ".json"]:
                archive.write(directory / name, f"features/{row['key']}/{name}")
            metadata = read_json(directory / "metadata.json")
            config = metadata.pop("config")
            # Local absolute paths are not part of contest deliverables.
            metadata["parameters"] = {k: v for k, v in config.items()
                                      if k not in {"data_root", "work_root", "model_provenance", "face_reviews"}
                                      and k not in {"mfa_dictionary", "mfa_acoustic", "openface_model_dir"}
                                      and not k.endswith(("_path", "_bin"))}
            archive.writestr(f"features/{row['key']}/metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.write(run / "summary.csv", "summary.csv")
        for path in (ROOT / "q1").glob("*.py"):
            archive.write(path, "source/q1/" + path.name)
        archive.write(ROOT / "README.md", "README.md")
        archive.write(ROOT / "requirements-server.txt", "source/requirements-server.txt")
        portable_config = read_json(ROOT / "configs" / "local.json")
        portable_config.update(data_root="data/raw100", work_root="work", mfa_bin="mfa",
                               openface_bin="FeatureExtraction", openface_model_dir="models/openface")
        archive.writestr("source/configs/local.json", json.dumps(portable_config, ensure_ascii=False, indent=2))
        runtime = read_json(run / "run.json")
        portable_runtime = dict(code=runtime["code"], environment=runtime["environment"],
                                models={k: {name: value for name, value in v.items() if name != "path"}
                                        for k, v in runtime["models"]["models"].items()},
                                external={k: {name: value for name, value in v.items() if name != "path"}
                                          if isinstance(v, dict) else v for k, v in runtime["external"].items()})
        archive.writestr("runtime_provenance.json", json.dumps(portable_runtime, ensure_ascii=False, indent=2))
    if target.stat().st_size > 50_000_000:
        oversized = target.with_suffix(".oversize.zip")
        target.replace(oversized)
        raise ValueError(f"Q1 archive exceeds 50 MB; retained for inspection at {oversized}")
    print("Q1 package:", target, "bytes:", target.stat().st_size,
          "(Q2/Q3 must share the competition's total attachment budget)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="MOSEI question 1; model stages are explicit, never auto-started")
    parser.add_argument("--config", default=str(ROOT / "configs" / "local.json"))
    parser.add_argument("--data-root")
    parser.add_argument("--work-root")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--face-reviews", help="JSON sample_id -> single_visible_speaker after visual review")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    sub.add_parser("plan")
    d = sub.add_parser("doctor")
    d.add_argument("--gpu", action="store_true")
    sub.add_parser("download-models")
    c = sub.add_parser("compare")
    c.add_argument("--sample", required=True)
    s = sub.add_parser("score")
    s.add_argument("--sample", required=True)
    s.add_argument("--reference", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--backend", choices=["whisperx", "mfa"], required=True)
    choose.add_argument("--sample", required=True)
    choose.add_argument("--reason", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--sample")
    b = sub.add_parser("bundle-inputs")
    b.add_argument("--output", required=True)
    pack = sub.add_parser("package")
    pack.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args)
    try:
        if args.command == "manifest":
            manifest = make_manifest(config["data_root"])
            write_json(Path(config["work_root"]) / "manifest.json", manifest)
            print(f"Verified {len(manifest['samples'])} samples; no models loaded")
            return 0
        if args.command == "plan":
            manifest = read_json(Path(config["work_root"]) / "manifest.json")
            ordinary = next(r for r in manifest["samples"] if r["video_id"] == "-3g5yACwYnA" and r["clip_id"] == "13")
            longest = max(manifest["samples"], key=lambda r: r["words"])
            print(json.dumps(dict(sample_count=len(manifest["samples"]), ordinary=ordinary["sample_id"],
                                  longest=longest["sample_id"], stages=["compare", "score optional manual boundaries",
                                  "select with evidence", "extract one sample", "extract all 100", "review/package"]), ensure_ascii=False, indent=2))
            return 0
        if args.command == "doctor":
            return doctor(config, args.gpu)
        if args.command == "download-models":
            download_models(config)
            return 0
        if args.command == "bundle-inputs":
            bundle(config, args.output)
            return 0
        manifest, run = context(config, require_models=True)
        print("Run:", run, flush=True)
        if args.command == "package":
            package_outputs(manifest, run, args.output)
            return 0
        if args.command == "extract":
            return extract_all(config, manifest, run, args)
        row = sample(manifest, args.sample)
        directory = run / "samples" / row["key"]
        if args.command == "compare":
            from .media import prepare
            from .aligners import Aligners
            from .report import compare
            media = prepare(config, row, directory)
            aligners = Aligners(config)
            results = {}
            for backend in ["whisperx", "mfa"]:
                results[backend] = aligners.run(backend, row, directory, media)
            print(json.dumps(compare(directory, row, results), ensure_ascii=False, indent=2))
        elif args.command == "score":
            from .report import score_reference
            results = {k: read_json(directory / f"alignment_{k}.json") for k in ["whisperx", "mfa"]}
            print(json.dumps(score_reference(directory, args.reference, results), indent=2))
        elif args.command == "select":
            comparison = read_json(directory / "comparison.json")
            if not args.reason.strip() or comparison[args.backend + "_coverage"] < config["alignment_coverage_min"]:
                raise ValueError("Selection requires a reason and adequate pilot alignment coverage")
            output = run / "selection.json"
            if output.exists() and read_json(output)["backend"] != args.backend:
                raise ValueError("Do not mix backends in one run. Use another --work-root for another selection")
            write_json(output, dict(backend=args.backend, reason=args.reason,
                                    pilot=row["sample_id"], run_signature=run.name,
                                    comparison_sha256=sha256(directory / "comparison.json")))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
