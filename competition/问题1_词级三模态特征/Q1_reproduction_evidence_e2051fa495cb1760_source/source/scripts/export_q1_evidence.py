"""Portable source, manifests, and command logs for the completed Q1 run."""
from pathlib import Path
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from q1.core import read_json,sha256,write_json

T=Path('/root/autodl-tmp/team')
run=Path(read_json(ROOT/'work/latest_run.json')['path'])
exp=T/'q1/exp/Q1-1_feature_extraction/outputs'
exp.mkdir(parents=True,exist_ok=True)
output=exp/f'Q1_reproduction_evidence_{run.name}.zip'
files=[]
for p in (ROOT/'q1').glob('*.py'): files.append((p,'source/q1/'+p.name))
for folder in ['configs','scripts']:
    files.extend((p,'source/'+p.relative_to(ROOT).as_posix()) for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for name in ['TEAM_RUNBOOK.md','Q1_METHOD_AND_OUTPUTS.md','requirements-server.txt']:
    p=ROOT/name
    if p.exists(): files.append((p,'source/'+name))
for p,name in [(ROOT/'work/manifest.json','input/manifest.json'),
               (ROOT/'models/provenance.json','models/provenance.json'),
               (run/'run.json','run/run.json'),(run/'selection.json','run/selection.json'),
               (run/'summary.csv','run/summary.csv'),
               (T/'README.md','protocol/server_rules.md'),
               (T/'shared/baselines/README.md','protocol/baselines_README.md'),
               (T/'reports/q1/Q1_METHOD_AND_OUTPUTS.md','reports/Q1_METHOD_AND_OUTPUTS.md'),
               (T/'reports/q1/Q1_visual_review.md','reports/Q1_visual_review.md'),
               (T/'q1/exp/Q1-2_alignment_validation/outputs/mfa-conda-explicit.txt','environment/mfa-conda-explicit.txt'),
               (T/'q1/exp/Q1-2_alignment_validation/outputs/mfa-pip-lock.txt','environment/mfa-pip-lock.txt'),
               (T/'q1/exp/Q1-1_feature_extraction/outputs/q1-package-versions.json','environment/q1-package-versions.json')]:
    if p.is_file(): files.append((p,name))
for p in run.glob('samples/*/*.log'):
    files.append((p,'run/'+p.relative_to(run).as_posix()))
for p in run.glob('samples/*/commands.log'):
    files.append((p,'run/'+p.relative_to(run).as_posix()))
for p in run.glob('samples/*/media.json'):
    files.append((p,'run/'+p.relative_to(run).as_posix()))
for p in (T/'q1/exp/Q1-1_feature_extraction/logs').glob('*q1.log'):
    files.append((p,'jobs/Q1-1/'+p.name))
for p in (T/'q1/exp/Q1-2_alignment_validation/logs').glob('*q1.log'):
    files.append((p,'jobs/Q1-2/'+p.name))
for p in (T/'q1/exp/Q1-3_att4_word_times/logs').glob('*q1.log'):
    files.append((p,'jobs/Q1-3/'+p.name))
for p in (T/'reports/q1/tables').glob('*.csv'):
    files.append((p,'reports/tables/'+p.name))
seen=set()
with zipfile.ZipFile(output.with_suffix('.zip.tmp'),'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path,arcname in sorted(files,key=lambda x:x[1]):
        if arcname in seen: continue
        seen.add(arcname)
        z.write(path,arcname)
    z.writestr('README.txt','Q1 reproduction evidence: source, verified input manifest, model/tool provenance, per-sample command/media logs, experiment job logs, and quality tables. Raw videos and model weights are omitted; their source SHA256 hashes and model hashes are recorded. The separate 100-sample research archive contains the feature arrays. Labels stored in metadata are identifiers/provenance only and must not be used as model input.\n')
output.with_suffix('.zip.tmp').replace(output)
write_json(exp/'Q1_reproduction_evidence.json',dict(path=str(output),sha256=sha256(output),bytes=output.stat().st_size,files=len(seen)))
print(output,output.stat().st_size,len(seen))
