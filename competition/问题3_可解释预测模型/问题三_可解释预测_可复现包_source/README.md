# 问题三：可解释多模态情感预测（轻量版 Self-MM）可复现包

本包包含问题三模型的**全部训练所得参数**、核心代码、配置、运行环境说明，以及附件4的预测与解释结果。唯一不在包内的是公开的预训练模型 `bert-base-uncased`：它在本模型中完全冻结、参数未作任何改动，加载时程序会核对其指纹，与训练时不一致即报错。

## 1. 目录

```
team/
  q3/exp/E3-25_frozen_bert_light/
    code/                 # 本问全部代码（见第 4 节）
    outputs/attn4/seed{1111,1112,1113}/
      compact.pt          # 除 BERT 外的全部参数（语音/视觉 LSTM、注意力池化、融合网络、单模态头），约 0.68 MB
      subset_head_fp16.pt # 子集感知融合头（半精度），约 0.24 MB
      subset_head.json    # 融合头的选择过程与验证集指标
      predictions/        # 验证集、测试集、附件4 的 8 子集预测
    tables/attn4/         # 附件4 证据、逐位置积分梯度、验证集删除检验、汇总
  shared/baselines/
    repos/MMSA/           # MMSA 框架（MIT 许可），Self-MM 实现
    common/               # 统一评测指标 metrics.py、Shapley 与区域分解 model_api.py、附件数据转换 convert_to_mmsa.py
    configs/mmsa/config_regression_att2.json
  q1/exp/Q1-3_att4_word_times/tables/att4_word_times.csv   # 问题一给出的附件4逐词元时间
results/
  attachment4_predictions_explanations.csv   # 附件4 预测与解释（提交文件）
  q3_numbers.json                            # 论文中引用的全部统计量
tools/q3_analysis.py                         # 由各种子预测生成统计量、表格、图与提交 CSV
MANIFEST_SHA256.json
```

## 2. 数据处理规则

- 使用附件2 **aligned_50** 版本，训练、验证、专项测试（附件4）保持同一版本和同一输入接口：`text_bert (3,50)`、`audio (50,74)`、`vision (50,35)`。
- 训练只用附件2 `train`（3395 条）；检查点、融合头的 λ 与轮次、中性阈值 θ 只用 `valid`（728 条）选择；`test`（727 条）只在定稿后评估一次。附件4 只做推理。
- 情感极性：连续输出 < −θ 为负向（0），> θ 为正向（2），其余为中性（1）；θ = 0.18，由验证集按加权 F1 在 0～0.8（步长 0.02）中选取。判为中性时输出强度为 0。
- 缺模态遮挡：文本正文词元换成 `[UNK]`（id 100，保留 `[CLS]`、`[SEP]` 与注意力掩码）；语音、视觉换成零向量。

## 3. 关键参数

| 部分 | 设置 |
|---|---|
| 文本 | bert-base-uncased，**全部冻结**；末四层隐状态平均，在正文词元上做注意力池化（768→1 打分器，769 个参数） |
| 语音 / 视觉 | 单层 LSTM 74→16、35→32，接线性层 |
| 融合 | 816→128→128→1（ReLU）；训练时另有文本、语音、视觉三个单模态回归头 |
| 训练目标 | Self-MM：多模态 L1 + ULGM 单模态伪标签的加权 L1 |
| 优化 | batch 32；学习率：语音 5e-3、视觉 1e-4、融合 1e-3、注意力池化 1e-3；按验证集损失早停，耐心 8 |
| 子集感知融合头 | 由融合网络初始化；损失 = 完整 MAE + λ·7 种子集平均 MAE，λ∈{0.1,0.25,0.5}；AdamW 2e-4，最多 60 轮，耐心 12 |
| 集成 | 种子 1111、1112、1113 等权平均 |
| 解释 | 8 子集精确 Shapley；容斥区域分解；积分梯度 128 步；证据片段取连续 3 个位置；删除检验对照所有同长度窗口 |

## 4. 运行步骤（Linux + NVIDIA GPU）

环境：Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.57.6、NumPy 2.3.2、Pandas 3.0.6；运行时 GPU 为 RTX 4080 SUPER。

1. **准备公开 BERT**：把 `bert-base-uncased` 放到 `team/shared/baselines/pretrained/bert-base-uncased/`。已核对的 `model.safetensors` 的 SHA256 前缀为 `68d45e234eb4a928`。
2. **转换赛题数据**：在 `team/shared/baselines/` 下设置 `Q3_DATA_ROOT` 为赛题原始数据目录，运行
   ```bash
   python common/convert_to_mmsa.py att2 --out data
   python common/convert_to_mmsa.py att4-export --out data   # 需 numpy>=2（附件4 原文件由 numpy 2 保存）
   python common/convert_to_mmsa.py att4-pack --out data
   ```
   得到 `data/mosei_att2_aligned50.pkl` 与 `data/mosei_att4_aligned50_infer.pkl`（附件4 的标签字段只是推理占位值，不参与任何计算）。
3. **只用包内参数复现附件4 结果**（约 5 分钟）：在 `team/q3/exp/E3-25_frozen_bert_light/` 下运行
   ```bash
   for s in 1111 1112 1113; do python code/subset_head.py --seed $s --pool attn4 --infer_only; done
   python code/attribution.py --pool attn4 --seeds 1111 1112 1113
   ```
   第一行用半精度融合头重算 8 子集预测，写到 `predictions_fp16/`；半精度与单精度的预测差不超过 3e-4。第二行重算积分梯度、证据与删除检验。
4. **从头训练**（约 20 分钟）：
   ```bash
   for s in 1111 1112 1113; do python code/train_light.py --seed $s --pool attn4; done
   python code/subset_head.py --seed 1111 --pool attn4
   bash code/run_chain.sh attn4      # 其余两个种子的融合头 + 归因
   ```
   GPU 浮点运算不保证逐位一致，重训结果与包内参数会有微小差异。
5. **生成统计量、表格与提交 CSV**：`python tools/q3_analysis.py`。脚本内的路径按本地目录设置，使用前请修改文件开头的 `SRC`、`PKG` 等常量。

## 5. 附件4 提交文件字段

| 字段 | 含义 |
|---|---|
| sample_id | 附件4 样本编号 01–20，与 `对齐版本/NN.pkl` 一一对应 |
| pred_polarity_code / pred_polarity | 0 negative / 1 neutral / 2 positive |
| pred_intensity | 提交的情感强度；中性为 0 |
| pred_score_raw | 置 0 之前的连续输出（所有解释针对此值） |
| phi_T / phi_A / phi_V | 三模态 Shapley 值，和等于完整输入与全遮挡预测之差 |
| share_T / share_A / share_V | 绝对份额 \|φ_m\| / Σ\|φ\| |
| main_modality | 主要参考模态；完整与全遮挡之差小于 0.1 时为 uncertain |
| max_share_modality | 份额最大的模态（main_modality 为 uncertain 时仍给出） |
| full_minus_empty | 完整输入与全遮挡预测之差 |
| interaction_type | additive_or_unique / synergy / redundant / conflict / indeterminate |
| U_T … C | 容斥区域 R_T、R_A、R_V、R_TA、R_TV、R_AV、R_TAV（定向） |
| text_evidence / text_token_span / text_evidence_sec | 文本证据片段、BERT 词元位置、原视频时间（秒） |
| audio_evidence_sec / audio_evidence_words | 语音证据时段及对应单词 |
| vision_keyframe_sec / vision_keyframe_frame / vision_evidence_sec | 视觉关键帧时间、帧号（30 帧/秒）、证据时段；样本 13 视觉特征全零，留空 |
| *_evidence_level | 删除检验分级：强 / 中 / 弱 |
| main_evidence_drop / main_evidence_random_drop | 主要模态证据片段的定向降幅，以及同长度随机片段的平均降幅 |

时间换算依据：附件4 对齐版第 p 行就是第 p 个 BERT 词元的特征，同一单词的子词特征完全相同；单词时间取自问题一对附件4 的强制对齐（`att4_word_times.csv`）。
