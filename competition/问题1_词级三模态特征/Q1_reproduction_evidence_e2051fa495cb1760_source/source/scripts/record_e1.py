"""Append one completed three-seed MMSA experiment to the team log."""
from pathlib import Path
import argparse
import csv
import fcntl

p=argparse.ArgumentParser()
p.add_argument('--model',choices=['mult','misa','self_mm','almt'],required=True)
a=p.parse_args()
root=Path('/root/autodl-tmp/team')
name=f'mmsa_{a.model}'
exp=root/'exp_common/E1_complete_baselines'/name
table=exp/'tables/summary_3seeds.csv'
with table.open(encoding='utf-8-sig') as f:
    rows={(r['split'],r['metric']):r for r in csv.DictReader(f)}
assert len(rows)==12
acc=rows['test','Acc3']; f1=rows['test','F1_3_weighted']; mae=rows['test','MAE']
readme=exp/'README.md'
content=readme.read_text(encoding='utf-8')
content=content.replace('状态 | 未开始（开始后改为“进行中 / 完成（日期）”）','状态 | 三种子训练与预测完成（2026-09-23）')
marker='\n<!-- Q1_E1_RESULTS -->'
content=content.split(marker)[0]+marker+f'''\n## 实际结果\n训练与逐样本预测：1111、1112、1113，三个种子均通过 728/727/30/20 条 ID 数量与唯一性检查。完整验证集和测试集的均值、样本标准差及逐种子数字见 tables/summary_3seeds.csv。测试集 Acc3={float(acc['mean']):.4f}±{float(acc['std']):.4f}，加权F1={float(f1['mean']):.4f}±{float(f1['std']):.4f}，MAE={float(mae['mean']):.4f}±{float(mae['std']):.4f}。\n按验证集保存唯一最佳权重，tau仅从验证集确定。检查点暂留在每个种子的 models/，供问题二 E2-1 缺失鲁棒性评估使用；后续完成依赖后依协作规范清理。\n'''
readme.write_text(content,encoding='utf-8')
entry_id=f'E1-{a.model.upper()}-3SEED'
log=root/'reports/EXPERIMENT_LOG.md'
with log.open('a+',encoding='utf-8') as f:
    fcntl.flock(f,fcntl.LOCK_EX)
    f.seek(0)
    if f'| {entry_id} |' not in f.read():
        f.write(f"| {entry_id} | 2026-09-23 | q1 | E1 {a.model}完整数据基线 | MMSA@a94e65d | scripts/team_e1_{a.model}.sh；附件2 aligned，BERT由模型配置控制 | 1111/1112/1113 | 验证集Loss选单一最佳检查点，验证集选tau | exp_common/E1_complete_baselines/{name}/tables/summary_3seeds.csv | 测试Acc3={float(acc['mean']):.4f}±{float(acc['std']):.4f}；F1(w)={float(f1['mean']):.4f}±{float(f1['std']):.4f}；MAE={float(mae['mean']):.4f}±{float(mae['std']):.4f} | 预测文件四划分已按ID核对；检查点供E2-1使用 |\n")
print(entry_id)
