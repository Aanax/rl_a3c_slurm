#!/bin/bash
set -euo pipefail

name_part="oracle_8_64_v3_relu_g_false"

/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle_8_64_v3_relu_g_false.ini \
  > "run_ids_${name_part}.txt"
