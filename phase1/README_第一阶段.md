**第一阶段：复现、序列排错与三种汇聚方式的受控比较**

本包在原提交材料以外独立运行。`baseline/` 是冻结的原始代码、压缩权重和校准文件；启动时核对 SHA-256。请勿直接执行其中的 `run.py train`：原训练入口会自动评估 test。本包的新入口不提供 test 评分功能，只在数据审计时检查 test 的尺寸与 ID。

**当前状态**

代码准备、合成数据单元检查与真实数据验证分开记录。未运行真实数据前，不宣称已复现原结果、已验证 12GB 显存足够或已经获得新科学结论。`verification_complete.json`、`training_complete.json`、`diagnostic_complete.json` 分别标识对应任务成功。

**服务器**

单张 RTX 3080 Ti 12GB 可先进行实测。默认训练 batch=8、评估 batch=8、BERT 序列长度取数据文件。第一轮只使用预提取特征，不重新运行视频、OpenFace 或 Whisper。首个 GPU smoke 记录三个版本的训练峰值显存；它不等于完整实验耗时。若 batch=8 不适配，应统一调整所有对照配置并重做验证，不在单一版本上临时改配置。

环境优先复用现有 PyTorch，建议另建隔离环境以免影响原工程：

```bash
/root/miniconda3/bin/python -m venv --system-site-packages /root/autodl-tmp/msa_phase1_env
/root/autodl-tmp/msa_phase1_env/bin/python -m pip install -r requirements.txt
```

已知原训练环境是 torch 2.3.0+cu121、Transformers 4.44.2；提交材料也记录过 torch 2.8.0 / Transformers 4.57.6 的指标复核。当前实例的 torch 2.8.0+cu128 能否复现，须由本包的复现门槛确认。CUDA 运行库版本与 nvidia-smi 显示的驱动支持上限不同，勿据此替换驱动。

依赖安装命令会在新环境中安装 NumPy 1.26.4、scikit-learn 1.5.2、Transformers 4.44.2，不主动替换已有 CUDA PyTorch。安装后先执行 `python -m unittest test_phase1 -v`。

**需要的数据**

1. 比赛原始附件二 `aligned_50.pkl`，原记录约 994 MB，3395/728/727；不能用其他 MOSEI 子集代替复现原成绩。
2. 与原权重匹配的 `bert-base-uncased` 本地目录，含 `config.json` 和参数文件。原压缩权重加载器核对冻结层指纹，不能绕过。
3. 第二批实验使用作者公开的 MOSI `aligned_50.pkl`，须含 `text_bert (N,3,L)` 和真正按同一 L 对齐的 audio、vision。本包支持音视频维度不同，不做未对齐到对齐的静默重采样。

MOSI 数据出处及文件哈希可核对 [MMSA 作者仓库](https://github.com/thuiar/MMSA#2-datasets)。仓库当前列出的 MOSI aligned_50.pkl SHA-256 是 `d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9`。这是作者公开文件的身份校验，不是竞赛附件二的哈希。

若原 BERT 不在，可从 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) 取得候选基础模型，再让原加载器验证；指纹不符时应寻找原基础权重。新实验也必须记录基础模型的文件哈希。

**执行顺序**

以下在本目录执行，把三个路径改成服务器实际位置。`PY`、`DATA`、`BERT`、`OUT` 仅是任务变量。

```bash
PY=/root/autodl-tmp/msa_phase1_env/bin/python
DATA=/实际路径/附件2/aligned_50.pkl
BERT=/实际路径/bert-base-uncased
OUT=/root/autodl-tmp/msa_phase1_results/contest_v1

# 检查数据和环境、复现原模型、导出原模型位置诊断、测三种版本显存。
$PY -u run_stage.py --stage verify --dataset contest --data "$DATA" --bert "$BERT" --output "$OUT"

# 查看 verification_complete.json、reproduction/reproduction.json、gpu_smoke/smoke.json。
# 通过后先跑 seed=42 的 last / mean / attention / text_only 四个版本。
$PY -u run_stage.py --stage pilot --dataset contest --data "$DATA" --bert "$BERT" --output "$OUT"

# 人工检查首轮后，才扩展 seed=43、44；不自动启动这一步。
$PY -u run_stage.py --stage multiseed --dataset contest --data "$DATA" --bert "$BERT" --output "$OUT"
```

同一输出目录不得覆盖已有训练。部分运行失败时，保留日志，按具体原因处理或换新的输出目录；不要删除失败证据后原地重复。

MOSI 先用自己的 DATA 和新 OUT 执行 `--dataset mosi --stage verify`，该分支不拿竞赛权重核对 MOSI 成绩；仍检查数据和 GPU。再执行 MOSI pilot。两数据集的指标不可混合求均值。

**各实验保持什么、改变什么**

- 三模态 last：与原模型结构一致，作为新一轮全精度重训的参考。
- 三模态 mean：音视频 LSTM 状态在已观测内容位置上求均值；其他部分不变。
- 三模态 attention：对相同状态做已观测位置注意力汇聚。每分支只加 hidden+1 个打分参数；零初始化时等价于 mean。hidden=128 时总共多 258 个参数，需在论文中披露。
- text_only：相同结构和训练流程，但声画输入与声画观测标记恒为零；这是输入消融，不是独立优化过的最强文本模型。

三个汇聚版本共享结构部分以相同种子初始化；原有 RNN 和输出投影不被随机替换。训练目标、ULGM、优化设置、六场景选轮次规则与原方案一致。使用独立完整权重，比较时不混入 INT8 存储压缩的误差。

当前只验证汇聚方式；无训练遮挡的交叉消融、其他骨干和真实媒体失真留到先导结果出来后决定。

**输出与解释边界**

- `preflight.json`：数据哈希、维度、有效位置比例、视频分组重叠、序列边界和环境。未知 ID 格式不冒充已完成视频分组核验。
- `reproduction/reproduction.json`：32 个验证场景与原表的逐指标误差，默认误差门槛 0.001；超过则停止继续训练。
- `gpu_smoke/smoke.json`：各版本 3 个优化步骤的峰值显存、参数量和耗时。
- `runs/*/epochs.jsonl`：每轮训练损失、选择分数、六场景指标、显存与耗时。
- `runs/*/best.pt`：仅由 valid 六场景选出的全精度检查点。
- `runs/*/calibration.json`：只在 valid 拟合且与检查点 SHA-256 绑定的校准。
- `runs/*/diagnostics/metrics.csv`、`samples.csv`：32 个原场景加四个整模态删除诊断。保存未校准分类、原始连续回归误差、校准后结果、逐样本变化和实际删去的已观测比例。
- `all_validation_metrics.csv`：完成的各版本结果，未自动判定假设成立。

raw_mae 使用原始连续预测；普通 mae 使用极性约束后的输出。需同时检查这两种口径，不能把中性归零的效果当作融合增益。所有诊断均为开发集探索结果。

整模态删除、局部遮挡都是受控扰动，可能产生训练分布外输入；预测变化只说明敏感性，必须结合预测是否更准确和只文本对照讨论。text_only 训练模型与三模态模型推理时删除 AV 是不同实验。

**首轮继续判断**

先确认数据序列、原模型复现和显存正常，再看汇聚变化是否改变位置敏感性、是否带来实际声画收益。之后在 MOSI 复查。若只发现具体实现错误，先修错重跑；若声画没有额外价值，不强行让贡献均等；若结果不支持原假设，保留负结果并收缩结论。

当前阶段不使用 test 成绩选方案；完整 MOSEI 与竞赛数据的重合关系另行审计。测试集此前曾被查看的事实应在最终论文中如实说明。
