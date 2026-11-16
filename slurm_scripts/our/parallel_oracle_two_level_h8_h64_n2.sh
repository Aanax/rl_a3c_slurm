#!/bin/bash
for i in $(seq $@)
do
 sbatch /s/ls4/users/aamore/rl_a3c_pytorch/slurm_scripts/our/run_one_run_oracle_two_level_h8_h64_n2.sh $i
done
