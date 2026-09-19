#!/bin/bash
set -euo pipefail

name_part="oracle_intrinsic_nomemory_16_advantage_v1"

/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle_intrinsic_nomemory_16_advantage_v1.ini \
  > "run_ids_${name_part}.txt"
