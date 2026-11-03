"""Audit generated arrays without changing scientific outputs or masks."""
from pathlib import Path
import sys
import collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from q1.core import read_json, write_json, write_csv
from q1.cli import verify_complete
run=Path(read_json(ROOT/'work/latest_run.json')['path'])
manifest=read_json(ROOT/'work/manifest.json')
report=Path('/root/autodl-tmp/team/reports/q1')
rows=[]
errors=[]
def covered_duration(intervals):
    total=0.0
    end=-float('inf')
    for a,b in sorted(intervals.tolist()):
        total+=max(0.0,b-max(a,end))
        end=max(end,b)
    return total
for r in manifest['samples']:
    d=run/'samples'/r['key']
    if not verify_complete(d):
        errors.append(dict(sample_id=r['sample_id'],issue='missing_or_invalid_completion'))
        continue
    try:
        with np.load(d/'aligned.npz',allow_pickle=False) as a, np.load(d/'native.npz',allow_pickle=False) as n:
            meta=read_json(d/'metadata.json')
            count=len(meta['words'])
            for name,dim in [('text',768),('audio',25),('vision',22)]:
                assert a[name].shape==(count,dim),(name,a[name].shape)
                assert np.isfinite(a[name]).all(),name
                assert a[name+'_mask'].shape==(count,),name
            bounds=a['word_intervals'][a['alignment_mask']]
            assert np.all(bounds[:,1]>bounds[:,0])
            assert np.all(bounds[1:,0]>=bounds[:-1,1]-1e-4)
            assert np.all(bounds>=0)
            assert np.all(bounds<=meta['media']['clip_duration']+1e-3)
            assert a['padding_mask'].all()
            assert np.all(a['audio'][~a['audio_mask']]==0)
            assert np.all(a['vision'][~a['vision_mask']]==0)
            assert len(n['video_pts'])==len(n['vision'])
            low=[w for w in meta['words'] if w['valid'] and (w['confidence'] is None or w['confidence']<.3)]
            rows.append(dict(sample_id=r['sample_id'],array_audit='passed',words=count,
                             low_confidence_words=len(low),low_confidence_text=' '.join(w['word'] for w in low),
                             audio_valid_duration_sec=covered_duration(n['audio_intervals'][n['audio_mask']]),
                             vision_valid_duration_sec=covered_duration(n['vision_intervals'][n['vision_mask']]),
                             **{k:v for k,v in meta['quality'].items() if k not in ['sample_id','words']},
                             warnings=';'.join(meta['warnings'])))
    except Exception as exc:
        errors.append(dict(sample_id=r['sample_id'],issue=str(exc)))
if rows:
    write_csv(report/'tables/Q1_array_audit.csv',rows)
    write_csv(report/'tables/T1-1_feature_summary.csv',rows)
write_json(report/'array_audit.json',dict(run=run.name,passed=len(rows),expected=100,errors=errors,
           total_words=sum(r['words'] for r in rows),low_score_words=sum(r['low_confidence_words'] for r in rows),
           low_vision_samples=sum(r['vision_word_coverage']<.7 for r in rows),
           low_alignment_samples=sum(r['alignment_coverage']<.95 for r in rows),
           note='Array integrity and timestamp availability are not manual accuracy validation. Score<0.3 is a diagnostic threshold, not calibrated correctness.'))
print(read_json(report/'array_audit.json'))
if errors:
    raise SystemExit(1)
