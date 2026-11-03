#!/usr/bin/env bash
# Q1-6：在新的工作目录中从头重跑问题一（同一代码、同一配置），再与 e2051fa495cb1760 批次逐项比较。
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PATH=/root/autodl-tmp/envs/q1-mfa/bin:$PATH
export MFA_ROOT_DIR=/root/autodl-tmp/mfa_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/root/autodl-tmp/mosei-q1/mosei_q1
EXP=/root/autodl-tmp/team/q1/exp/Q1-6_reproduction
W=$EXP/work/q1_rerun
PY=/root/autodl-tmp/envs/q1/bin/python
cd "$P"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
$PY -m q1 --data-root data/raw100 --work-root "$W" manifest
$PY -m q1 --data-root data/raw100 --work-root "$W" compare --sample 94d7ec6188bed1e2
$PY -m q1 --data-root data/raw100 --work-root "$W" select --backend whisperx --sample 94d7ec6188bed1e2 --reason 'Reproduction of run e2051fa495cb1760 with identical code and configuration; same provisional choice as the original run.'
$PY -m q1 --data-root data/raw100 --work-root "$W" extract
RUN=$($PY -c "import json;print(json.load(open('$W/latest_run.json'))['path'])")
$PY scripts/compare_runs.py --a work/runs/e2051fa495cb1760 --b "$RUN" --out "$EXP/tables"
