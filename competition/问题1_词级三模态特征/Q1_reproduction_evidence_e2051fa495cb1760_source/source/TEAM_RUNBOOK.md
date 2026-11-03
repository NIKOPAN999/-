# 协作服务器实际运行记录（2026-09-23）

本记录优先于旧的单人服务器运行示例。先阅读工作区《服务器操作规范.md》及 baselines/README.md。所有实验通过 team/tools/launch.sh 登记；q1 原目录保持不动，shared 只读。MFA/OpenFace 运行前通知协作群；本次用户已确认通知。

## 实際环境

- q1：Python 3.12.3，torch/torchaudio 2.8.0+cu128，WhisperX 3.8.6，transformers 4.57.6，numpy 2.3.2，openSMILE 2.6.0，openpyxl 3.1.5，soundfile 0.13.1，huggingface-hub 0.36.0。
- q1-mfa：Python 3.11，MFA 3.4.2，kalpy 0.10.5，Kaldi 5.5.1172；额外补充 jinja2 3.1.6。MFA 3.3.1 的 generate_features 接口与 kalpy 0.10.5 不兼容，不再使用该组合。
- MFA 程序及 Kaldi 工具依赖 q1-mfa/bin 在 PATH 中；特征脚本始终显式使用 q1/bin/python。
- WhisperX 只使用已有转写的 alignment 模块；当前安装是最小依赖路径，不包含 diarization。requirements-server.txt 是早期完整安装方案，不是本次的精确锁文件。实际包版本保存在 Q1-1/outputs/q1-package-versions.json，MFA 锁文件在 Q1-2/outputs。
- OpenFace 使用 OpenCV 4.5.4、dlib 19.10；构建时放宽了 dlib 最低版本检查。源码为下载的 master ZIP，不把它冒称为固定 commit；ZIP SHA256 为 3ee54f94e8b702b0afcd5b66217601229b7e9644dcb8f72d6e27c59bdd11e62b。
- CEN 模型来自 kabachuha/openface_cen_patches 镜像。run.json 记录实际可执行文件、人脸模型和 AU_predictors 全部哈希；存在来源/构建差异，需在复现说明中保留。

## 已运行的入口

在 /root/autodl-tmp/team 中执行，实验目录已存在。不要直接后台运行 Python，不要使用 pkill。

```bash
ENV=q1 bash tools/launch.sh q1 q1/exp/Q1-2_alignment_validation 'bash /root/autodl-tmp/mosei-q1/mosei_q1/scripts/team_compare.sh'
ENV=q1 bash tools/launch.sh q1 q1/exp/Q1-1_feature_extraction 'bash /root/autodl-tmp/mosei-q1/mosei_q1/scripts/team_extract.sh'
ENV=q1 bash tools/launch.sh q1 q1/exp/Q1-3_att4_word_times 'bash /root/autodl-tmp/mosei-q1/mosei_q1/scripts/team_att4.sh'
```

每个任务设置 OMP/MKL/OPENBLAS 线程数为 4。先看 jobs.sh；当前代码在每条特征输出通过哈希核验后只删除该条自己的临时 PNG，保留原视频、native.npz、aligned.npz 和时间戳。输出中 True 表示可用；缺失视觉数据为零向量且掩码为 False，不能把零向量作为真实视觉观测。

team_finalize.sh 是本次登记任务的收尾器，等待当前提取 PID 结束后运行汇总、数组审计和研究备份；脚本中的 PID 是本次任务专属，后续不要直接复用。重新运行批处理时，需从同配置的 selection 和 complete.json 校验续跑。

## 结果口径

单样本15个词，两种对齐均有时间。平均边界差异258.77ms、P95为1197ms，不能解释成真值误差。非约束CTC诊断在句首识别到原稿外语音，对 They've 的位置约为1.44秒，与WhisperX一致；这是同一声学模型的辅助证据，仍非人工准确率验证。

暂选WhisperX用于批处理，保持原稿，绝不把识别结果替换为标签。BERT768、eGeMAPSv02的25维LLD、选定22维OpenFace特征按重叠时长聚合。不是MMSA-FET默认74/35维特征的数值复现，不可直接套入附件2的74/35维模型。

附件4按提供的text_bert逐位置核对，保留全部1000个位置：506个词汇词元有时间，40个特殊位、396个填充位、58个非词汇标点位置无时间（-1）。24个词元分数低于0.3，另表列出。0.3只是工程复核阈值，不是经过标注验证的正确率阈值。问题三应排除无时间位置，低分词元先复核再用作证据。

抽帧已观察到风景、鸟类、产品、字幕遮挡、侧脸等情形，部分低人脸覆盖有真实数据原因。当前speaker_identity_not_reviewed不代表代码失败，也不代表已确认单一画面人物就是说话人。不要为了通过打包检查直接清除告警。

研究备份zip不是最终竞赛附件；最终打包还须完成质量复核并满足三问合计50MB等赛题要求。实际进度和质量结果见 results/q1 下从服务器同步的报告。
