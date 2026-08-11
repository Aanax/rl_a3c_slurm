#!/bin/sh
#SBATCH -D /s/ls4/users/aamore/rl_a3c_pytorch/
#SBATCH -o /s/ls4/users/aamore/rl_a3c_pytorch/logs/%j.out
#SBATCH -e /s/ls4/users/aamore/rl_a3c_pytorch/logs/%j.err
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -p hpc4-el7-gpu-3d

export CUDA_HOME=/s/ls4/sw/cuda/10.2/
export LD_LIBRARY_PATH="/s/ls4/sw/cuda/10.2/lib64:$LD_LIBRARY_PATH"

echo "CUDA EXPORTED"

module load intel-compilers cuda/10.2

source /s/ls4/users/aamore/anaconda3/bin/activate
conda activate pytorch_rl

echo "ACTIVATED"

param=$1
python /s/ls4/users/aamore/rl_a3c_pytorch/src/main.py /s/ls4/users/aamore/rl_a3c_pytorch/configs/run_config_our.ini ${param} | tee mytask_logs/mytask.log."$SLURM_JOBID"

