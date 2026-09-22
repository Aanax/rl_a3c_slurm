#!/bin/bash
set -euo pipefail

name_part="oracle_two_level_pacman"

bash /s/ls4/users/aamore/rl_a3c_pytorch/slurm_scripts/our/parallel_oracle_runner.sh "$@" \
  "${name_part}" \
  /s/ls4/users/aamore/rl_a3c_pytorch/configs/run_config_oracle_two_level_pacman.ini
