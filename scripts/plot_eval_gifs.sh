#!/usr/bin/env bash
set -euo pipefail

EVAL_FOLDER="Eval_2026-04-23_19:07:36_1024fc_A3CRules2378OracleIntrinsicCritic_32w_8_64_v2_1"
USERNAME="dartl0l"
REMOTE_PATH="/home/users/dartl0l/goarl/rl_a3c_slurm/logs/eval"
SERVER="${SERVER:-ui4-el7.computing.kiae.ru}"
SSH_KEY="${SSH_KEY:-~/.ssh/dartl0l_key}"

python3 src/draw_eval_gifs.py \
  "${EVAL_FOLDER}" \
  --username "${USERNAME}" \
  --remote-path "${REMOTE_PATH}" \
  --server "${SERVER}" \
  --ssh-key "${SSH_KEY}" \
  "$@"
