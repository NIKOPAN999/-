"""Read-only source audit, copies Q1 results to collaboration report directories."""
from pathlib import Path
import collections
import csv
import json
import shutil
import sys
import fcntl
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q1.core import read_json, write_json, write_csv
from q1.cli import verify_complete

T = Path('/root/autodl-tmp/team')
run = Path(read_json(ROOT / 'work/latest_run.json')['path'])
manifest = read_json(ROOT / 'work/manifest.json')
pilot = run / 'samples/94d7ec6188bed1e2'
exp2 = T / 'q1/exp/Q1-2_alignment_validation'
exp1 = T / 'q1/exp/Q1-1_feature_extraction'
exp3 = T / 'q1/exp/Q1-3_att4_word_times'
report = T / 'reports/q1'
for d in [report/'tables', report/'figures', report/'pilot']:
    d.mkdir(parents=True, exist_ok=True)
for filename in ['comparison.csv', 'comparison.json', 'alignment_review.html', 'manual_reference.csv', 'alignment_whisperx.json', 'alignment_mfa.json', 'audio.wav', 'greedy_ctc_diagnostic.json']:
    if (pilot/filename).is_file():
        shutil.copy2(pilot/filename, report/'pilot'/filename)
        shutil.copy2(pilot/filename, exp2/('tables' if filename.endswith('.csv') else 'outputs')/filename)
records = []
for row in manifest['samples']:
    d = run/'samples'/row['key']
    if verify_complete(d):
        c = read_json(d/'complete.json')
        records.append(dict(sample_id=row['sample_id'], status=c['status'], **{k:v for k,v in c['quality'].items() if k!='sample_id'}, warnings=';'.join(c['warnings'])))
summary = dict(run=str(run), completed=len(records), expected=len(manifest['samples']),
               failures=[read_json(p) for p in (run/'samples').glob('*/failure.json') if not verify_complete(p.parent)],
               review_flags=dict(collections.Counter(w for r in records for w in r['warnings'].split(';') if w)),
               timestamp=datetime.now(timezone.utc).isoformat())
if records:
    write_csv(exp1/'tables/extraction_progress.csv', records)
    write_csv(report/'tables/Q1_feature_quality.csv', records)
write_json(report/'progress.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
for source in (exp3/'tables').glob('*.csv'):
    shutil.copy2(source, report/'tables'/source.name)

message = f'''# 问题一阶段结果

- 当前批次 `{run.name}`：已核验 {len(records)}/100 条特征输出。
- BERT 768 维、openSMILE eGeMAPSv02 25 维、OpenFace 22 维；按词时间窗与原始帧时间窗的重叠时长聚合。
- 单样本两种对齐覆盖率均为 15/15；平均边界差异 258.77 ms，P95 为 1197 ms。无人工真值，不能据此宣称某工具更准确。
- 暂选 WhisperX：可复用模型并提供词级声学分数；句首大分歧保留回放复核。
- 同一声学模型的非约束贪心解码在 0.1–0.56 秒识别出原稿外语音，在约 1.44 秒识别出 They've；支持 WhisperX 的句首定位，但不是独立模型或人工真值。
- 附件4时间戳：20 条、1000 个位置（含特殊/填充位），与原 text_bert 严格逐项匹配。无时间位置为 -1；不能把特殊/填充位当作证据片段。
- 附件4有 506 个带时间的词汇词元，其中 24 个分数低于 0.3，见 att4_low_confidence_tokens.csv；阈值仅供复核筛选，时间覆盖率不代表准确率。
- 视觉使用单人跟踪，speaker_identity_not_reviewed 表示尚未完成逐视频说话人核验，不能视为已人工验收。
- 环境修复：MFA 3.3.1 与 kalpy 0.10.5 不兼容，问题一专用环境升级至 MFA 3.4.2，并补充 jinja2 3.1.6。未修改共享 msa 环境。
- OpenFace 使用现有本地构建；dlib 版本检查放宽、CEN 权重来自镜像，须在复现实验中注明，不能声称完全复现 MMSA-FET 默认配置。

数据表在 tables/，单样本回放和两套时间戳在 pilot/。
'''
(report/'q1_pipeline_status.md').write_text(message, encoding='utf-8')
conclusions=report/'conclusions.md'
existing=conclusions.read_text(encoding='utf-8')
marker='\n<!-- Q1_ACTUAL_RESULTS -->'
conclusions.write_text(existing.split(marker)[0]+marker+'\n'+message+'\n固定预训练提取与对齐不套用训练实验的三随机种子均值要求；没有生成虚构种子结果。\n',encoding='utf-8')
index=T/'reports/paper/figures_tables_index.md'
content=index.read_text(encoding='utf-8')
if len(records)==100:
    content=re.sub(r'^\| T1-1 \|.*$', '| T1-1 | 附件1 100条样本特征及质量统计 | 表 | q1/tables/T1-1_feature_summary.csv | Q1-100 | q1 | 自动提取完成，人工复核待处理 | 问题一（2） |',content,flags=re.MULTILINE)
if '| F1-1 |' not in content:
    content+='\n| F1-1 | 单样本WhisperX/MFA词边界对照（非真值误差） | 图 | q1/figures/Q1_pilot_alignment.png | Q1-PILOT | q1 | 自动对照完成 | 问题一（3） |\n'
index.write_text(content,encoding='utf-8')
for exp, status, detail in [(exp1, '进行中' if len(records)<100 else '特征生成完成，质量复核待处理',f'已校验 {len(records)}/100，批次 {run.name}；完整质量表见 tables/extraction_progress.csv'),
                            (exp2,'完成自动对照，人工边界待复核','15/15 vs 15/15，平均差异 258.77 ms，不是准确率。'),
                            (exp3,'完成自动对齐','20 条全部完成，词元 ID 与位置完全一致；见 tables/att4_word_times.csv。')]:
    p=exp/'README.md'
    content=p.read_text(encoding='utf-8')
    content=re.sub(r'\| 状态 \|[^\n]*', f'| 状态 | {status} |', content, count=1)
    marker='\n<!-- Q1_PIPELINE_STATUS -->'
    content=content.split(marker)[0]
    p.write_text(content+marker+f'\n## 当前实际状态\n{status}。{detail}\n',encoding='utf-8')

# Append completed result records only, under a lock; never rewrite team log.
with (T/'reports/EXPERIMENT_LOG.md').open('a+',encoding='utf-8') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    f.seek(0); content=f.read()
    entries=[('Q1-MFA-342','MFA 环境修复','MFA3.4.2/kalpy0.10.5','旧版3.3.1 API不兼容及缺少jinja2已修复','q1/exp/Q1-2_alignment_validation/logs/','版本检查和单样本通过'),
             ('Q1-PILOT','WhisperX/MFA 单样本对照','WhisperX3.8.6/MFA3.4.2','scripts/team_compare.sh','q1/exp/Q1-2_alignment_validation/tables/comparison.csv','两者15/15，平均差异258.77ms，未测准确率'),
             ('Q1-ATT4','附件4词元时间戳','WhisperX3.8.6','scripts/team_att4.sh','q1/exp/Q1-3_att4_word_times/tables/att4_word_times.csv','20条，词元位置完全匹配；自动时间戳待按需人工复核')]
    if len(records)==100:
        entries.append(('Q1-100','100条三模态特征','BERT/openSMILE/OpenFace','scripts/team_extract.sh','reports/q1/tables/Q1_feature_quality.csv','100/100生成完成，视觉身份复核未完成'))
    for key,title,model,cmd,path,result in entries:
        if f'| {key} |' not in content:
            f.write(f'| {key} | 2026-09-23 | q1 | {title} | {model} | {cmd} | 不适用（预训练特征/对齐） | 未使用情感标签选模 | {path} | {result} | 自动结果，参见实验README |\n')
