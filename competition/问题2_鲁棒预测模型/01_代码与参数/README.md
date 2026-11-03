# 问题二局部缺失情感预测模型复现说明

本目录包含问题二代码、定型压缩权重、匹配校准文件、训练日志和主模型场景指标。正式预测文件位于相邻目录`02_附件3预测结果/附件3_情感极性与强度预测.csv`。

## 输入与使用边界

- 输入为赛题**原始**附件2 `aligned_50.pkl` 和附件3的`对齐版本/附件3_01.pkl`至`附件3_30.pkl`。原始数据由赛题提供，不在提交包中。不能用团队内部转换的PKL代替。
- 附件2 train学习网络和标准化统计量；valid选择轮次并拟合校准；test用于已定型模型的报告；无标签附件3只用于最终推理。
- 文本用BERT编码，语音、视觉用单层单向LSTM。观测标记与覆盖率表示特征可用性；训练时随机加入局部连续遮挡；单模态辅助头和ULGM伪标签参与原模型训练。输出经验证集校准和符号一致性投影。
- INT8/FP16用于**参数文件压缩存储**。加载时反量化为浮点模型；本实现没有验证INT8算子推理加速。

## 软件和基础权重

原实验记录为PyTorch `2.3.0+cu121`、Transformers `4.44.2`、NumPy `1.26.4`、scikit-learn `1.5.2`、CUDA `12.1`；原Python版本未记录。先准备兼容的PyTorch，再执行`python -m pip install -r requirements.txt`。

还必须准备与训练一致的公开`bert-base-uncased`基础权重（含`config.json`及参数文件）。压缩权重省略了冻结的embedding、前8层和pooler；程序以`SELFMM_BERT_BASE`指定本地目录，并核对其冻结参数SHA-256指纹：

`8cba3c662753f601da0336fb0cb9abdbc3623f7dbeba1d2f620958db2ee4b613`

原始下载revision未保存。指纹不符时应换用经核对的基础权重，不应跳过检查。

## 用已提交权重复核

在**本目录**打开终端，按本机位置修改下列三个路径。命令显式传入数据路径；`settings.py`的默认路径保留原开发布局，不能依赖它自动找到当前数据。

```bash
export SELFMM_BERT_BASE='/绝对路径/bert-base-uncased'
DATA='/绝对路径/附件2-数据集特征文件/aligned_50.pkl'
ATT3='/绝对路径/附件3-模态缺失特征样本/对齐版本'

python -u run.py evaluate --data "$DATA" --checkpoint student_compact.pt --calibration student_compact_calibration.json --split valid --output-csv valid_reproduced.csv
python -u run.py evaluate --data "$DATA" --checkpoint student_compact.pt --calibration student_compact_calibration.json --split test --output-csv test_reproduced.csv
python -u run.py infer --checkpoint student_compact.pt --calibration student_compact_calibration.json --attachment3 "$ATT3" --output-csv attachment3_reproduced.csv
```

校准文件与权重通过SHA-256绑定。原始输入四指标按Accuracy、macro F1、MAE、Pearson排序，valid预期为`0.6648 / 0.6311 / 0.5600 / 0.6663`，test为`0.7111 / 0.6596 / 0.5962 / 0.6926`。前两条命令各写出32种场景，可与`compact_valid_calibrated.csv`、`compact_test_calibrated.csv`核对；第三条生成30行预测，可与正式附件3 CSV逐行比较。跨硬件浮点末位差异应单独报告。

## 从头训练与证据边界

`settings.py`及`training_conditions.json`包含原训练设置。准备相同基础权重后运行：

```bash
python -u run.py train --data "$DATA" --bert-name "$SELFMM_BERT_BASE" --output outputs/retrain_01
```

脚本不会覆盖本目录定型权重。原训练主种子为42，遮挡专用生成器的种子为`42+17=59`；batch 8，最长20轮，早停耐心4轮。`training.jsonl`和`epoch_metrics.csv`只保存定型方案的一次训练轨迹。其他消融版本的原始记录仍未全部提供；因此不能仅凭现有附件完整重算全部消融结论。

## 逐样本与配对区间复核

`../03_复跑与不确定性证据/`含定型模型在官方附件2的valid/test逐样本结果、valid按视频分组的配对自助抽样程序和结果、valid/test的32场景复跑表、附件3的独立复跑CSV，以及双块和三模态错位缺失压力测试的规则与验证结果。逐样本文件只包含原始输入和文本末段50%遮挡两场景，是论文6.8节的配对区间的证据，不代表附件3有真实标签。`../04_关键消融复跑/`另含同环境重训的full、no_mask、no_aux三版本各三个种子的配置、日志和32场景valid原始记录；该目录README明确区分定型模型、早期探索性汇总与此次复跑。导出命令示例：

```bash
python -u export_sample_evidence.py --data "$DATA" --checkpoint student_compact.pt --calibration student_compact_calibration.json --split valid --output-csv ../03_复跑与不确定性证据/valid_sample_evidence.csv
python -u export_sample_evidence.py --data "$DATA" --checkpoint student_compact.pt --calibration student_compact_calibration.json --split test --output-csv ../03_复跑与不确定性证据/test_sample_evidence.csv
```

在证据目录运行`python bootstrap_check.py`，可按239个valid视频组、种子20260925、2000次重复重算论文6.8节的配对区间；该程序仅依赖NumPy。复跑使用的PyTorch/Transformers版本与原训练环境不同，因此逐场景浮点末位可能变化，四位小数指标一致。

| 文件 | 内容 |
|---|---|
| `run.py`、`data.py`、`model.py`、`ulgm.py`、`calibration.py`、`settings.py` | 核心代码与参数 |
| `student_compact.pt`、`student_compact_calibration.json` | 最终推理参数与校准 |
| `epoch_metrics.csv`、`training.jsonl` | 主模型训练轨迹 |
| `compact_valid_calibrated.csv`、`compact_test_calibrated.csv`、`compact_valid_test_metrics.csv` | 主模型场景指标 |
| `environment.json`、`training_conditions.json`、`requirements.txt` | 环境与训练条件 |
| `export_sample_evidence.py`、`../03_复跑与不确定性证据/` | 逐样本复跑和6.8节配对区间证据 |
| `../04_关键消融复跑/` | 三版本×三种子同环境配置、训练轨迹、32场景valid对照与消融结果绘图程序（论文6.7节图表由论文作者重新绘制） |

方法来源：Yu等，*Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis*，AAAI 2021，DOI `10.1609/aaai.v35i12.17289`。本实现采用Self-MM思路，并非逐行复现公开代码。
