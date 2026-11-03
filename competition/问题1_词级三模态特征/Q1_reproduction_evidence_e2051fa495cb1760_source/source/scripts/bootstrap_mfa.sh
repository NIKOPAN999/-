#!/usr/bin/env bash
set -euo pipefail
export CONDA_PKGS_DIRS=/root/autodl-tmp/conda_pkgs
ENV=/root/autodl-tmp/envs/q1-mfa
/root/miniconda3/bin/conda create -p "$ENV" --override-channels -c conda-forge python=3.11 'montreal-forced-aligner=3.4.2' 'kalpy=0.10.5' 'ffmpeg>=6,<8' -y
export PATH="$ENV/bin:$PATH"
export MFA_ROOT_DIR=/root/autodl-tmp/mfa_cache
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
mfa version
/root/miniconda3/bin/conda list -p "$ENV" --explicit > /root/autodl-tmp/mosei-q1/mosei_q1/mfa-environment-explicit.txt
printf 'MFA_ENV_COMPLETE\n'
