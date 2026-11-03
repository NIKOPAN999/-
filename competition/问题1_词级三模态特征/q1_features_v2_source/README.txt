问题一：100 条原始样本的词级三模态时序特征（v2）

samples/<样本编号>/          目录直接以样本编号命名（编号中的 $_$ 为原样）
  aligned_v2.npz            词级特征：text(L×768)、audio(L×25，按参数组分别聚合，基频/共振峰组只用浊音帧)、vision(L×22)、
                            word_intervals(L×2，无效为[-1,-1])、alignment_state(2可靠/1存疑/0无效)、alignment_score、
                            audio_energy_mask、audio_voiced_mask、vision_mask、motion_mask、voiced_ratio、prosody(L×6)、
                            face_absent_ratio、low_conf_ratio、motion、mfa_iou(-1=MFA无结果)、asr_hit、
                            gap_intervals/gap_type/gap_features/gap_masks（非词片段）、utt_*（句级特征）
  metadata_v2.json          三态统计、转写一致性、画面核查结论、特征列名、聚合参数与约定
  native.npz                原生帧特征与时间：openSMILE LLD、OpenFace 22维、区间、有效标记、置信度、视频帧PTS
  metadata.json             原文、子词映射、媒体信息、模型与工具版本
  alignment_whisperx.json   WhisperX 原始对齐输出
  complete.json             各文件 SHA256
  frame_quality.npz         逐视频帧：跟踪成功、置信度、画面运动能量（灰度帧差）
tables/                     全量 WhisperX-MFA 逐词对照、无约束识别、跨硬件复现比较、帧质量统计
code/                       aggregate_v2.py（词级聚合）、mfa_full.py、asr_check.py、openface_quality.py、compare_runs.py、
                            team_repro.sh、export_att2_format.py（按附件2对齐版结构导出）、q1_revision_analysis.py（论文统计量）
summary_v2.json、speaker_review.json、run.json、selection.json、summary.csv

按附件2对齐版结构导出的 q1_aligned50.pkl 受附件体积限制未放入，可用 code/export_att2_format.py 由本包重新生成。

读取示例（无需 pickle）：
  import numpy as np; a = np.load("samples/-3g5yACwYnA$_$13/aligned_v2.npz", allow_pickle=False)
零向量不代表观测值，读取时须结合对应掩码与 alignment_state。
