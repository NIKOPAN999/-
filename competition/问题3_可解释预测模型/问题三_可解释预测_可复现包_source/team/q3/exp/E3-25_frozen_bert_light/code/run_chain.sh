#!/usr/bin/env bash
# 等种子 1112、1113 训练完成后，依次训练子集融合头，再做三种子集成的位置归因与删除检验
set -euo pipefail
POOL=${1:-attn4}
for s in 1112 1113; do
  until [ -f outputs/$POOL/seed$s/compact_info.json ]; do sleep 15; done
  python code/subset_head.py --seed $s --pool $POOL
done
python code/attribution.py --pool $POOL --seeds 1111 1112 1113
echo CHAIN_DONE
