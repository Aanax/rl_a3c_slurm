#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <seq_end | seq_start seq_end> <job_name> <config_path>"
  exit 1
fi

job_name="${@: -2:1}"
config_path="${!#}"
seq_args=("${@:1:$(($#-2))}")

for i in $(seq "${seq_args[@]}")
do
  sbatch -J "${job_name}_${i}" /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/slurm_scripts/our/run_one_run_oracle.sh "$i" "$config_path"
done
