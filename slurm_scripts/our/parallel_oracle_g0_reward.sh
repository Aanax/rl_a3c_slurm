#!/bin/bash
set -euo pipefail

name_part="oracle_go_reward"

/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle_g0_reward.ini \
  > "run_ids_${name_part}.txt"
