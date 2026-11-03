"""Independent descriptive summary and paired video-cluster bootstrap for phase 1."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

VERSIONS=('last','mean','attention','text_only')
SEEDS=(42,43,44)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    all_records=[];paired={};details={}
    for cohort,n in (('contest',728),('mosi',229)):
        models={};details[cohort]={}
        source=read_csv(args.root/(cohort+'_pilot_summary.csv'))
        for version in VERSIONS:
            runs=[];clean=[]
            for seed in SEEDS:
                folder=args.root/f'{cohort}_v1/runs/{version}_s{seed}'
                assert (folder/'diagnostics/diagnostic_complete.json').is_file()
                metric={r['scenario']:r for r in read_csv(folder/'diagnostics/metrics.csv')}
                sample=[r for r in read_csv(folder/'diagnostics/samples.csv') if r['scenario']=='clean']
                assert len(sample)==n
                ids=[r['sample_id'] for r in sample];groups=[r['video_id'] for r in sample]
                if clean:assert ids==clean[0]['ids'] and groups==clean[0]['groups']
                y=np.array([float(r['true_intensity']) for r in sample])
                cls=np.array([int(r['true_class']) for r in sample])
                pred=np.array([float(r['raw_score']) for r in sample])
                pc=np.array([int(r['raw_class']) for r in sample])
                accuracy=float(np.mean(pc==cls));mae=float(np.mean(abs(pred-y)))
                assert abs(accuracy-float(metric['clean']['raw_accuracy']))<1e-10
                assert abs(mae-float(metric['clean']['raw_mae']))<1e-7
                clean.append({'ids':ids,'groups':groups,'y':y,'class':cls,'pred':pred,'pred_class':pc})
                runs.append({'seed':seed,'accuracy':accuracy,'mae':mae,
                             'macro_f1':float(metric['clean']['raw_macro_f1']),
                             'calibrated_accuracy':float(metric['clean']['accuracy']),
                             'calibrated_macro_f1':float(metric['clean']['macro_f1']),
                             'calibrated_mae':float(metric['clean']['mae']),
                             'drop_text_accuracy':float(metric['drop_text']['raw_accuracy']),
                             'drop_text_mae':float(metric['drop_text']['raw_mae']),
                             'drop_av_accuracy':float(metric['drop_audio_vision']['raw_accuracy']),
                             'drop_av_mae':float(metric['drop_audio_vision']['raw_mae'])})
            models[version]=clean
            row={'cohort':cohort,'version':version,'n_seeds':3,'n_valid':n,
                 'n_video_groups':len(set(clean[0]['groups']))}
            for key in ('accuracy','mae','macro_f1','calibrated_accuracy',
                        'calibrated_macro_f1','calibrated_mae','drop_text_accuracy',
                        'drop_text_mae','drop_av_accuracy','drop_av_mae'):
                values=[r[key] for r in runs]
                row[key+'_mean']=float(np.mean(values));row[key+'_sd']=float(np.std(values,ddof=1))
            sr=[r for r in source if r['run'].startswith(version+'_s')]
            assert len(sr)==3
            for mode in ('audio','vision'):
                for pos in ('front','end'):
                    key=f'{mode}_half_{pos}_score_change'
                    row[key]=float(np.mean([float(r[key]) for r in sr]))
                    metric_name=f'{mode}_0.5_{pos}'
                    field=f'mean_removed_observed_{mode}_fraction'
                    fractions=[]
                    for seed in SEEDS:
                        rows=read_csv(args.root/f'{cohort}_v1/runs/{version}_s{seed}/diagnostics/metrics.csv')
                        fractions.append(float(next(r for r in rows if r['scenario']==metric_name)[field]))
                    row[f'{mode}_half_{pos}_removed_fraction']=float(np.mean(fractions))
            all_records.append(row);details[cohort][version]=runs
        reference=models['text_only']
        group_labels=reference[0]['groups'];unique=sorted(set(group_labels))
        group_index=np.array([unique.index(g) for g in group_labels])
        counts=np.bincount(group_index,minlength=len(unique))
        draws=np.random.default_rng(20261029).integers(0,len(unique),size=(2000,len(unique)))
        paired[cohort]={}
        for version in VERSIONS[:-1]:
            acc=[];mae=[]
            for i in range(3):
                a=models[version][i];b=reference[i]
                assert a['ids']==b['ids']
                np.testing.assert_array_equal(a['y'],b['y'])
                acc.append((a['pred_class']==a['class']).astype(float)-(b['pred_class']==b['class']).astype(float))
                mae.append(abs(b['pred']-b['y'])-abs(a['pred']-a['y']))
            record={}
            for name,values,scale in (('accuracy_gain_pp',acc,100.),('raw_mae_reduction',mae,1.)):
                per_sample=np.mean(values,axis=0)
                sums=np.bincount(group_index,weights=per_sample,minlength=len(unique))
                boot=sums[draws].sum(1)/counts[draws].sum(1)
                record[name]={'mean':float(per_sample.mean()*scale),
                              'video_bootstrap_ci95':(np.quantile(boot,[.025,.975])*scale).tolist(),
                              'per_seed':(np.mean(values,axis=1)*scale).tolist()}
            paired[cohort][version]=record
    result={'summary':all_records,'paired_comparison_to_text_only':paired,'per_seed':details,
            'scope':'Exploratory validation only; no test evaluation. Confidence intervals resample original videos conditional on the three trained seeds; they do not remove model-selection bias or quantify all training randomness.',
            'bootstrap_repetitions':2000,'bootstrap_seed':20261029}
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_records[0]));w.writeheader();w.writerows(all_records)
    print(json.dumps({'completed_runs':24,'summary_rows':len(all_records),
                      'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':main()
