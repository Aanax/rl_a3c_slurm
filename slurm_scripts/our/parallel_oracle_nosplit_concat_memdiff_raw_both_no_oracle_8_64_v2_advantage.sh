#!/bin/bash
set -euo pipefail

name_part="oracle_nosplit_concat_memdiff_raw_both_no_oracle_8_64_v2_advantage"

/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle_nosplit_concat_memdiff_raw_both_no_oracle_8_64_v2_advantage.ini \
  > "run_ids_${name_part}.txt"
