#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PATH=/root/autodl-tmp/envs/q1-mfa/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
E=/root/autodl-tmp/team/q1/exp/Q1-3_att4_word_times
mkdir -p "$E/work/videos"
tar -xzf "$E/work/att4_videos.tar.gz" -C "$E/work/videos"
/root/autodl-tmp/envs/q1/bin/python /root/autodl-tmp/mosei-q1/mosei_q1/scripts/att4_word_times.py --videos "$E/work/videos" --features /root/autodl-tmp/team/shared/baselines/data/mosei_att4_aligned50_infer.pkl --output "$E"
