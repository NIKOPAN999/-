"""Fetch public MOSI features and a pinned BERT; verify content before use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

MOSI_SHA256="d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9"
BERT_REVISION="86b5e0934494bd15c9632b12f734a8a67f723594"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def fetch(url,path,expected=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file():
        actual=digest(path)
        if expected and actual!=expected:
            raise ValueError(f"Existing file SHA-256 mismatch: {path}")
        return {"path":str(path),"url":url,"bytes":path.stat().st_size,"sha256":actual,"reused":True}
    part=path.with_name(path.name+".part")
    start=time.perf_counter()
    with requests.get(url,stream=True,timeout=(20,60)) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:f.write(chunk)
    actual=digest(part)
    if expected and actual!=expected:
        raise ValueError(f"Downloaded content SHA-256 mismatch: {part}; retained for inspection")
    part.rename(path)
    result={"path":str(path),"url":url,"bytes":path.stat().st_size,"sha256":actual,
            "elapsed_seconds":time.perf_counter()-start,"reused":False}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--endpoint",default="https://hf-mirror.com")
    args=p.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    endpoint=args.endpoint.rstrip("/")
    model_info=requests.get(endpoint+f"/api/models/google-bert/bert-base-uncased/tree/{BERT_REVISION}",timeout=30)
    model_info.raise_for_status()
    tree=model_info.json()
    model_expected=next(r["lfs"]["oid"] for r in tree if r["path"]=="model.safetensors")
    jobs=[(endpoint+"/datasets/tamb2203579/CMU-MOSI/resolve/main/Processed/aligned_50.pkl?download=true",
           root/"MOSI/aligned_50.pkl",MOSI_SHA256)]
    for name in ("config.json","model.safetensors","tokenizer_config.json","tokenizer.json","vocab.txt"):
        jobs.append((endpoint+f"/google-bert/bert-base-uncased/resolve/{BERT_REVISION}/{name}?download=true",
                     root/"bert-base-uncased"/name,model_expected if name=="model.safetensors" else None))
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(lambda j:fetch(*j),jobs))
    provenance={"dataset":"CMU-MOSI aligned feature file mirrored on Hugging Face",
                "author_reference":"https://github.com/thuiar/MMSA#2-datasets",
                "author_published_mosi_sha256":MOSI_SHA256,
                "mirror_repository":"tamb2203579/CMU-MOSI",
                "bert_repository":"google-bert/bert-base-uncased","bert_revision":BERT_REVISION,
                "bert_expected_weights_sha256":model_expected,"files":records}
    (root/"asset_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    print("ASSETS_READY",root,flush=True)


if __name__=="__main__":main()
