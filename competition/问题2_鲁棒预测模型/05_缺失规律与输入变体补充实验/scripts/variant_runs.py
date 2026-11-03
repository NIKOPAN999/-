"""问题二补实验：缺失感知输入单因素消融与文本单模态对照（三种子）。

在队员定型代码的副本上只改一处开关（环境变量 Q2_VARIANT，由本脚本写入副本的 model.py），其余训练、
选轮次、校准流程与队员附件04 的 full/no_mask/no_aux 完全相同：
  text_only   : 语音、视觉的观测标记全部置 False（输入被清零，模态表示为 0，覆盖率为 0），只用文本
  no_coverage : 融合层不再接收三路覆盖率（以 0 代替，维度不变）
  no_obsflag  : 语音、视觉 LSTM 不再接收逐位观测标记（该通道以 0 代替；缺失位置特征仍为 0）
用法：python variant_runs.py text_only        （依次跑种子 42、43、44）
训练结束只保留配置、日志与 valid 32 场景结果；删除权重与 test 结果（test 不用于补实验）。
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "work/code_base"
DATA = Path("/root/autodl-tmp/team/shared/baselines/data/mosei_att2_aligned50.pkl")
BERT = Path("/root/autodl-tmp/team/shared/baselines/pretrained/bert-base-uncased")
variant = sys.argv[1]
assert variant in ("text_only", "no_coverage", "no_obsflag")

PATCH_MODEL = {
    "text_only": ('        observed = batch["observed"].bool()\n',
                  '        observed = batch["observed"].bool().clone()\n'
                  '        observed[:, :, 1:] = False  # E2-13 text_only: audio/vision unavailable\n'),
    "no_coverage": ('        fusion = self.fusion(torch.cat((text, audio, vision, coverage), dim=-1))\n',
                    '        fusion = self.fusion(torch.cat((text, audio, vision, torch.zeros_like(coverage)), dim=-1))  # E2-13 no_coverage\n'),
    "no_obsflag": ('                            observed.unsqueeze(-1).to(features.dtype)), dim=-1)\n',
                   '                            torch.zeros_like(observed.unsqueeze(-1).to(features.dtype))), dim=-1)  # E2-13 no_obsflag\n'),
}

env = os.environ.copy()
env["SELFMM_BERT_BASE"] = str(BERT)
for seed in (42, 43, 44):
    name = f"{variant}_s{seed}"
    work, out = ROOT / "work" / name, ROOT / "outputs" / name
    assert not out.exists(), name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for p in BASE.glob("*.py"):
        shutil.copy2(p, work / p.name)
    for p in list(BASE.glob("*.txt")) + list(BASE.glob("*.md")):
        shutil.copy2(p, work / p.name)
    m = (work / "model.py").read_text(encoding="utf-8")
    old, new = PATCH_MODEL[variant]
    assert m.count(old) == 1, (variant, "patch anchor not unique")
    (work / "model.py").write_text(m.replace(old, new), encoding="utf-8")
    s = (work / "settings.py").read_text(encoding="utf-8")
    assert '"seed": 42' in s
    (work / "settings.py").write_text(s.replace('"seed": 42', f'"seed": {seed}'), encoding="utf-8")
    print(f"=== START {name} ===", flush=True)
    subprocess.run([sys.executable, "-u", "run.py", "train", "--data", str(DATA), "--bert-name", str(BERT),
                    "--output", str(out)], cwd=work, env=env, check=True)
    assert (out / "compact_valid_calibrated.csv").is_file()
    for f in list(out.glob("*.pt")) + list(out.glob("*.zip")) + list(out.glob("*test*")):
        f.unlink()
    shutil.copy2(work / "model.py", out / "model_variant.py")
    shutil.copy2(work / "settings.py", out / "settings.py")
    print(f"=== DONE {name}; weights and test outputs removed ===", flush=True)
