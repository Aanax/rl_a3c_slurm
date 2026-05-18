#!/usr/bin/env bash
set -euo pipefail

USERNAME="dartl0l"
REMOTE_PATH="/home/users/dartl0l/goarl/rl_a3c_slurm/logs/eval"
SERVER="${SERVER:-ui4-el7.computing.kiae.ru}"
SSH_KEY="${SSH_KEY:-~/.ssh/dartl0l_key}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <eval_folder> [draw_eval_gifs.py options]"
  exit 1
fi

EVAL_FOLDER="$1"
shift

python3 src/draw_eval_gifs.py \
  "${EVAL_FOLDER}" \
  --username "${USERNAME}" \
  --remote-path "${REMOTE_PATH}" \
  --server "${SERVER}" \
  --ssh-key "${SSH_KEY}" \
  "$@"
