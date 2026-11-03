#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
B=/root/autodl-tmp/team/shared/baselines
PY=/root/autodl-tmp/envs/msa/bin/python
for seed in 1111 1112 1113; do
  "$PY" "$B/configs/mmsa/run_mmsa.py" --model mult --tag e1 --seeds "$seed" --aligned --set use_bert=true use_finetune=false --results_root outputs
  "$PY" "$B/configs/mmsa/predict_mmsa.py" --model mult --tag e1 --seed "$seed" --results_root outputs
done
