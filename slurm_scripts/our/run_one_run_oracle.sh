#!/bin/sh
#SBATCH -D /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/
#SBATCH -o /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/logs/%j.out
#SBATCH -e /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/logs/%j.err
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH -p hpc5-el7-gpu-3d

export CUDA_HOME=/s/ls4/sw/cuda/10.2/
export LD_LIBRARY_PATH="/s/ls4/sw/cuda/10.2/lib64:$LD_LIBRARY_PATH"

echo "CUDA EXPORTED"

module load intel-compilers cuda/10.2

source /s/ls4/users/dartl0l/anaconda/bin/activate
conda activate goarl_old

echo "ACTIVATED"

param=$1
config_path=${2:-/s/ls4/users/dartl0l/goarl/rl_a3c_slurm/configs/run_config_oracle.ini}
python /s/ls4/users/dartl0l/goarl/rl_a3c_slurm/src/main.py "${config_path}" "${param}" | tee mytask_logs/mytask.log."$SLURM_JOBID"
