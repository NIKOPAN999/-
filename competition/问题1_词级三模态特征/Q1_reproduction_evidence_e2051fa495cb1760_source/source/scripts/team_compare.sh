#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PATH=/root/autodl-tmp/envs/q1-mfa/bin:$PATH
export MFA_ROOT_DIR=/root/autodl-tmp/mfa_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /root/autodl-tmp/mosei-q1/mosei_q1
/root/autodl-tmp/envs/q1/bin/python -m q1 --data-root data/raw100 compare --sample 94d7ec6188bed1e2
