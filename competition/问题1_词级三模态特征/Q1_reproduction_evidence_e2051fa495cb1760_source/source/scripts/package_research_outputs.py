"""Backup verified research outputs; this is not the final competition submission."""
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from q1.core import read_json,write_json,sha256
from q1.cli import verify_complete
run=Path(read_json(ROOT/'work/latest_run.json')['path'])
rows=read_json(ROOT/'work/manifest.json')['samples']
assert len(rows)==100
assert all(verify_complete(run/'samples'/r['key']) for r in rows),'Not all 100 samples are complete'
output=Path('/root/autodl-tmp/team/q1/exp/Q1-1_feature_extraction/outputs')
archive=output/f'q1_features_{run.name}_research.zip'
with zipfile.ZipFile(archive.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
    z.write(run/'run.json','run.json')
    z.write(run/'selection.json','selection.json')
    z.write(run/'summary.csv','summary.csv')
    for row in rows:
        d=run/'samples'/row['key']
        for name in ['aligned.npz','native.npz','metadata.json','complete.json','alignment_whisperx.json']:
            z.write(d/name,f'samples/{row["key"]}/{name}')
    z.writestr('REVIEW_REQUIRED.txt','Research backup, not final competition submission. Preserve missingness masks. Speaker identities and uncertain word boundaries require review; do not interpret availability as alignment accuracy.')
archive.with_suffix('.zip.tmp').replace(archive)
write_json(output/'research_archive.json',dict(path=str(archive),sha256=sha256(archive),bytes=archive.stat().st_size))
print(archive)
