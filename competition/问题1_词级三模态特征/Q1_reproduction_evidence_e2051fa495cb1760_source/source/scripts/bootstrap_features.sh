#!/usr/bin/env bash
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
TASK=/root/autodl-tmp/mosei-q1/mosei_q1
ENV=/root/autodl-tmp/envs/q1
mkdir -p /root/autodl-tmp/envs
/root/miniconda3/bin/python -m venv --system-site-packages "$ENV"
"$ENV/bin/python" -m pip install --prefer-binary torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/python" -m pip install --prefer-binary -r "$TASK/requirements-server.txt"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$TASK/requirements-resolved.txt"
"$ENV/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.ones(4,device="cuda").sum().item())'
printf 'FEATURE_ENV_COMPLETE\n'
