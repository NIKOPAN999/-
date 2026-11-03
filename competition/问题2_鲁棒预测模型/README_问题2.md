# 问题二：局部缺失下的鲁棒情感预测模型

运行说明以`01_代码与参数/README.md`为准，数据处理规则与参数见`01_代码与参数/参数与数据规则.md`。

| 目录 | 内容 | 论文位置 |
|---|---|---|
| `01_代码与参数/` | 核心代码（`run.py`、`data.py`、`model.py`、`ulgm.py`、`calibration.py`、`settings.py`）；定型压缩权重`student_compact.pt`与校准文件（SHA-256绑定）；训练轨迹`epoch_metrics.csv`、`training.jsonl`；验证集与测试集32场景指标；`environment.json`、`requirements.txt`、`training_conditions.json` | 6.1～6.5节 |
| `02_附件3预测结果/` | 附件3的30条预测（与根目录`附件3_预测结果.csv`相同） | 6.9节 |
| `03_复跑与不确定性证据/` | 定型模型在验证集、测试集上的逐样本复跑结果；按原视频分组的配对自助抽样程序与结果；附件3独立复跑 | 6.8节 |
| `04_关键消融复跑/` | 完整模型、去掉训练遮挡、去掉辅助监督三个版本各三个种子的配置、训练轨迹与32场景结果 | 6.7节 |
| `05_缺失规律与输入变体补充实验/` | 定型模型只推理的缺失规律分析（剂量—反应、位置、单段与多段形态）；去掉覆盖率输入、去掉逐位观测标记、只用文本三个变体各三个种子的训练记录与结果 | 6.6.1～6.6.3、6.7节 |
| `06_基线对比实验/` | 与MulT、MISA、ALMT、Self-MM、EMT-DLFR、TFR-Net、LNLN在同一组随机同步缺失场景下的对比：导出脚本与全部逐种子结果 | 6.6.4节 |

## 快速核验（使用已提交权重，不训练）

```bash
cd 01_代码与参数
export SELFMM_BERT_BASE='/绝对路径/bert-base-uncased'
python -u run.py evaluate --data '/绝对路径/aligned_50.pkl' --checkpoint student_compact.pt --calibration student_compact_calibration.json --split valid --output-csv valid_reproduced.csv
python -u run.py infer --checkpoint student_compact.pt --calibration student_compact_calibration.json --attachment3 '/绝对路径/附件3对齐版本目录' --output-csv attachment3_reproduced.csv
```

验证集原始输入的四项指标应为 Accuracy 0.6648、macro-F1 0.6311、MAE 0.5600、Pearson 0.6663；第二条命令生成的30行结果应与`附件3_预测结果.csv`一致。

## 说明

- 训练只用附件2训练集；训练轮次、类别偏置与强度校准只用验证集确定；测试集只在定型后评估一次；附件3只用于推理。
- `05`、`06`中的脚本按团队服务器的目录布局编写，路径需按本机位置修改；`06`所用的基线权重体积过大，未随附件提交。
- 各子目录README中若出现图表编号，均指队员分稿中的编号；本文对应位置以上表的节号为准。
