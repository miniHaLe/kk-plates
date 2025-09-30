#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/hale/hale
yes y | exec /home/hale/.venv/bin/python /home/hale/hale/run.py --mode dashboard
