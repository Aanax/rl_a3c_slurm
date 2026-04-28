#!/bin/bash
for i in $(seq $@)
do
 sbatch /s/ls4/users/aamore/rl_a3c_pytorch/slurm_scripts/base/run_one_run_pacman.sh $i
done
