"""Validate three E1 MMSA seeds and aggregate validation/test metrics."""
from pathlib import Path
import argparse
import csv
import json
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('--model',choices=['mult','misa','self_mm','almt'],required=True)
args=parser.parse_args()
root=Path('/root/autodl-tmp/team/exp_common/E1_complete_baselines')/f'mmsa_{args.model}'
by_seed={}
for seed in (1111,1112,1113):
    d=root/'outputs/mmsa'/args.model/'e1'/f'seed{seed}'
    data=json.loads((d/'metrics.json').read_text())
    assert data['seed']==seed and data['model']==args.model
    for split,count,key in [('valid',728,'id'),('test',727,'id'),('att3',30,'sample_id'),('att4',20,'sample_id')]:
        with (d/'predictions'/f'{split}.csv').open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==count and len({r[key] for r in rows})==count,(seed,split,len(rows))
    by_seed[seed]=data
metrics=['Acc3','F1_3_weighted','Has0_Acc2','Non0_Acc2','MAE','Pearson']
rows=[]
for split in ('valid','test'):
    for metric in metrics:
        values=[by_seed[seed][split][metric] for seed in by_seed]
        rows.append(dict(model=args.model,split=split,metric=metric,n_seeds=3,
                         mean=statistics.mean(values),std=statistics.stdev(values),
                         seed1111=values[0],seed1112=values[1],seed1113=values[2]))
table=root/'tables/summary_3seeds.csv'
table.parent.mkdir(parents=True,exist_ok=True)
with table.open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(json.dumps(dict(model=args.model,tau={s:by_seed[s]['tau_selected_on_valid'] for s in by_seed},
                      summary=str(table),test={r['metric']:[r['mean'],r['std']] for r in rows if r['split']=='test'}),indent=2))
