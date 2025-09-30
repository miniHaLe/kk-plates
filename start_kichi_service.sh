#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
# giả lập env để run.py khỏi cảnh báo
export CONDA_DEFAULT_ENV=kichi
# nếu app có check thêm CONDA_PREFIX thì set luôn (không bắt buộc)
export CONDA_PREFIX="/home/toanbt/miniconda3/envs/kichi"

# auto trả lời "y" cho prompt "Continue anyway? (y/N):"
printf 'y\n' | exec /home/toanbt/miniconda3/envs/kichi/bin/python /home/hale/hale/run.py --mode dashboard
