# 问题一：词级三模态时序特征

| 文件 | 内容 | 论文位置 |
|---|---|---|
| `q1_features_v2.zip` | 附件1全部100条样本的特征文件。`samples/<样本编号>/`下为`aligned_v2.npz`（词级文本768维、语音25维、视觉22维特征，词区间，三态标记，各参数组掩码，词内动态，非词片段与句级特征）、`native.npz`（原生帧特征与时间戳）、`metadata.json`/`metadata_v2.json`（原文、子词映射、特征列名、媒体信息、转写一致性与画面核查结论）、`alignment_whisperx.json`、`frame_quality.npz`与`complete.json`（文件摘要）；`tables/`为全量对照与核查表；`code/`为聚合、对照、核查与统计脚本。包内`README.txt`给出全部字段 | 5.5、5.7节，附录A、B |
| `Q1_reproduction_evidence_e2051fa495cb1760.zip` | 复现证据：源代码（`source/`）、输入清单与原视频SHA-256（`input/`）、模型与工具来源（`models/`、`environment/`）、逐样本命令与媒体日志（`run/`）、任务日志（`jobs/`）与核查表（`reports/`） | 5.9节，附录C |
| `附件4_词级时间戳.csv` | 附件4共20条×50个BERT词元位置的时间：`sample_id, token_pos, wordpiece, word_idx, word, start_sec, end_sec, align_conf, status`，非词汇位置时间记为−1 | 5.8节；问题三的证据时间定位（7.5节） |

## 读取方式

数组均为NPZ格式，读取时不需要pickle：

```python
import json, numpy as np
d = "samples/-3g5yACwYnA$_$13"
a = np.load(f"{d}/aligned_v2.npz", allow_pickle=False)
X_t, X_a, X_v = a["text"], a["audio"], a["vision"]      # L×768, L×25, L×22
B, state = a["word_intervals"], a["alignment_state"]     # L×2 词区间；2可靠/1存疑/0无效
face = a["vision_mask"] & (state > 0)                     # 视觉特征可用的词
meta = json.load(open(f"{d}/metadata_v2.json", encoding="utf-8"))["info"]
```

零向量不代表观测值，必须结合对应掩码与`alignment_state`使用。

## 复现

1. 按论文附录C的工具版本准备环境（FFmpeg 7.1.1、WhisperX 3.8.6与wav2vec2-base-960h、openSMILE 2.6.0、OpenFace源码编译版、bert-base-uncased）。
2. 完整运行流程与每一步的命令见论文附录C的流程表，脚本位于复现证据包的`source/`；逐样本实际执行的命令与输出记录在`run/`与`jobs/`。
3. 词级特征可由`native.npz`中的原生帧特征直接重新聚合（特征包`code/aggregate_v2.py`），不需要重新提取。
4. 跨硬件重跑的比较结果见特征包`tables/`中的复现比较表：1934个词边界全部一致，语音、视觉词级特征与掩码逐元素一致，BERT文本特征只有浮点末位差异。
