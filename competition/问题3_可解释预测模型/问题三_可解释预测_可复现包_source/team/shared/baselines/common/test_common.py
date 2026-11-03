# -*- coding: utf-8 -*-
"""共用模块自检：python test_common.py（只需 numpy；读取 ../data 下转换好的 pkl）"""
import os, pickle, numpy as np
import missing, metrics

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "data", "mosei_att2_aligned50.pkl"), "rb") as f:
    valid = pickle.load(f)["valid"]

# 1) 同步伯努利：缺失率≈rate，三模态位置一致，[CLS]/[SEP] 不被遮
s = missing.bernoulli_sync(valid, 0.2, seed=1111)
r = missing.observed_missing_rate(s)
print("sync  r=0.2  观测缺失率 T/A/V = %.3f %.3f %.3f" % (r["T"].mean(), r["A"].mean(), r["V"].mean()))
assert np.array_equal(s["missing_mask"]["T"], s["missing_mask"]["A"])
assert (s["text_bert"][:, 0, 0] == valid["text_bert"][:, 0, 0]).all()           # [CLS] 保留
L = valid["audio_lengths"].astype(int)
assert all(s["text_bert"][i, 0, L[i]-1] == valid["text_bert"][i, 0, L[i]-1] for i in range(len(L)))  # [SEP] 保留
# 2) 异步：三模态掩码不同
a = missing.bernoulli_async(valid, 0.2, seed=1111)
print("async r=0.2  T 与 A 掩码一致的比例 = %.3f（应远小于 1）" % np.mean([np.array_equal(a["missing_mask"]["T"][i], a["missing_mask"]["A"][i]) for i in range(len(L))]))
# 3) 连续段：句尾遮 30%
sp = missing.span(valid, pos=1.0, frac=0.3, modalities=("T",))
rs = missing.observed_missing_rate(sp)
print("span pos=1.0 frac=0.3 只遮文本  观测 T/A/V = %.3f %.3f %.3f" % (rs["T"].mean(), rs["A"].mean(), rs["V"].mean()))
# 4) 原数据未被修改
assert (valid["text_bert"][:, 0, :] == 100).sum() == 0
# 5) 指标：真值自评应满分；常数预测 Pearson 为 nan
y = valid["regression_labels"]
e = metrics.evaluate(y, y, tau=0.0)
print("自评: Acc3=%.3f F1_3=%.3f MAE=%.3f Pearson=%.3f" % (e["Acc3"], e["F1_3_weighted"], e["MAE"], e["Pearson"]))
assert e["Acc3"] == 1 and e["MAE"] == 0
noisy = y + np.random.RandomState(0).normal(0, 0.6, size=y.shape)
tau, best = metrics.select_tau(y, noisy)
print("加噪预测: 验证集选出 tau=%.2f, F1_3=%.3f；tau=0.1 时 %s" % (tau, best, {k: round(v, 3) for k, v in metrics.evaluate(y, noisy, 0.1).items() if k in ("Acc3", "Has0_Acc2", "Non0_Acc2", "MAE", "Pearson")}))
print("全部检查通过")

# 6) model_api：用带交互项的假模型检验 Shapley 效率性与韦恩总量
import model_api
class Toy(model_api.BaseWrapper):
    name = "toy"
    def predict(self, sp):
        t = (sp["text_bert"][:, 0, 1:10] != 100).mean(1)          # 文本保留比例
        a = np.abs(sp["audio"]).mean((1, 2)) / 10
        v = np.abs(sp["vision"]).mean((1, 2))
        return 2 * t - 0.5 * a + 0.3 * v + 0.8 * t * v - 0.4 * a * v
g = model_api.subset_game(Toy(), valid)
r = model_api.venn_and_shapley(g)
full, empty = g[frozenset("TAV")], g[frozenset()]
eff = np.abs(sum(r["phi"].values()) - (full - empty)).max()
venn_total = sum(r["U"].values()) + sum(r["P"].values()) + r["C"]
tot = np.abs(venn_total - r["sign"] * (full - empty)).max()
print("model_api: 8 个子集，Shapley 效率性误差 %.1e，韦恩总量误差 %.1e" % (eff, tot))
assert eff < 1e-9 and tot < 1e-9
print("model_api 检查通过")
