# E2-14_baseline_compare

| 项目 | 内容 |
|---|---|
| 负责人 | q2 |
| 状态 | 2026-09-26 完成 |
| 目的 | 论文问题二对比实验：把论文中的问题二定型模型（队员提交的 student_compact.pt）与普通多模态基线、专门处理缺失的基线放在同一组缺失场景、同一评估口径下比较 |
| 依赖 / 输入 | E1 检查点（MISA / MulT / ALMT，各 1111–1113）；E2-11 已导出的 Self-MM 预测（`E2-11_selfmm_protocol/outputs/selfmm_s*.npz`）；E2-9 已导出的 EMT-DLFR（三种子）、TFR-Net、LNLN（单种子）预测（`E2-9_robust_arch/outputs/*/views.npz`）；附件2 对齐版 valid |
| 产出 | `outputs/team_views.npz`、`outputs/{misa,mult,almt}_s*.npz`；`tables/e214_views.csv` |
| 对应论文位置 | 问题二 6.7 节“与其他模型的对比” |

## 场景与口径（与 E2-9、E2-11 完全相同，不新定义）
- 场景：完整输入、三模态同步随机缺失 10%/20%/30%/50%（种子 1111，与附件3机制相同）、异步 20%，由 `bernoulli_split` 生成。
- 评估：E2-9 的 `eval_views.py`，按原视频两折交叉拟合 ×5；在拟合折的完整输入上选回归仿射校准与中性死区 τ；极性：回归型模型用死区，论文模型用自身分类头；输出做极性—强度一致化。
- 只推理：不训练、不选模、不读 test 与附件3。

## 脚本
- `predict_mmsa_views.py`：由 E2-11 的 `predict_selfmm_views.py` 改写，只加 `--model`。
- `predict_team_views.py`：论文模型在同一组场景上的未校准回归值与 logits；`bernoulli_split` 逐字照抄 code2，避免与队员 `data.py` 重名。

## 注意
- 表中论文模型的数值来自交叉拟合口径，与论文表18（验证集整体校准）略有差别。
- TFR-Net、LNLN 的检查点已在 E2-9 删除，只有单种子预测。

## 结果（2026-09-26）

逐种子结果见 e214_views.csv；论文表20、图15由 论文/tools/q2_compare.py 生成。论文模型在各缺失率下 Accuracy 最高；缺失率 0→0.5 下降 9.6 个百分点，与 EMT-DLFR（9.7）相当，普通基线下降 14.3～21.3 个百分点。
