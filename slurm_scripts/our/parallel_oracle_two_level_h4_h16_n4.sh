#!/bin/bash
for i in $(seq $@)
do
 sbatch /s/ls4/users/aamore/rl_a3c_pytorch/slurm_scripts/our/run_one_run_oracle_two_level_h4_h16_n4.sh $i
done
