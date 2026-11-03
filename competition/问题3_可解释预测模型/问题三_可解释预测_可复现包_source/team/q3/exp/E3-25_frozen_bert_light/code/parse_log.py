"""从 MMSA 日志中取验证 Loss 最低一轮的验证指标。"""
import re, sys
for path in sys.argv[1:]:
    lines = open(path, encoding='utf-8', errors='ignore').read().splitlines()
    best = None
    for i, l in enumerate(lines):
        m = re.search(r'VAL-\(self_mm\) >> loss: ([\d.]+)', l)
        if m:
            mm = re.search(r'MAE: ([\d.]+)\s+Corr: ([\d.]+)', lines[i + 1])
            ep = sum(1 for x in lines[:i] if 'VAL-(self_mm)' in x) + 1
            if best is None or float(m.group(1)) < best[0]:
                best = (float(m.group(1)), float(mm.group(1)), float(mm.group(2)), ep)
    n = sum(1 for x in lines if 'VAL-(self_mm)' in x)
    print(path, 'best valid loss %.4f MAE %.4f Corr %.4f at epoch %d of %d' % (*best, n))
