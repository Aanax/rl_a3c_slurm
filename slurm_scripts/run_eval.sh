#!/bin/sh
#SBATCH --job-name=rl_eval
#SBATCH -D /s/ls4/users/aamore/rl_a3c_slurm/
#SBATCH -o /s/ls4/users/aamore/rl_a3c_slurm/logs/%j.out
#SBATCH -e /s/ls4/users/aamore/rl_a3c_slurm/logs/%j.err
#SBATCH -t 00:30:00
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -p hpc4-el7-gpu-3d

# Usage:
#   sbatch slurm_scripts/run_eval.sh <config> <model_path> [additional_args]
#
# Arguments:
#   $1  path to .ini run config  (e.g. configs/run_config_our.ini)
#   $2  path to .dat/.pth checkpoint  (e.g. trained_models/PongNoFrameskip-v4.dat)
#   $3+ additional arguments for eval.py (e.g. --num-episodes 3 --gpu-id 0)
#
# Output:
#   Evaluation artifacts in logs/{experiment_name}/Eval_{timestamp}_{model_name}/
#   SLURM stdout/stderr in logs/%j.out / logs/%j.err
#   Tee'd log in logs/eval.${SLURM_JOBID}.log

CONFIG="${1}"
MODEL_PATH="${2}"

export CUDA_HOME=/s/ls4/sw/cuda/10.1/
export LD_LIBRARY_PATH="/s/ls4/sw/cuda/10.1/lib64:$LD_LIBRARY_PATH"

# >>> conda initialize >>>
__conda_setup="$('/s/ls4/users/aamore/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/s/ls4/users/aamore/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/s/ls4/users/aamore/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/s/ls4/users/aamore/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export PATH="/s/ls4/users/aamore/anaconda3/bin:$PATH"
source /s/ls4/users/aamore/anaconda3/bin/activate pytorch_rl
conda activate new_torch

echo "python: $(which python)"
echo "conda env: $(conda info --envs | grep '*')"
echo "------------------------------------------------------"
echo "CONFIG     : ${CONFIG}"
echo "MODEL_PATH : ${MODEL_PATH}"
echo "------------------------------------------------------"

# Build eval command with absolute paths
EVAL_CMD="python /s/ls4/users/aamore/rl_a3c_slurm/src/eval.py --config ${CONFIG} --model-path ${MODEL_PATH} --on-cluster"

# Add any additional arguments (from $3 onwards)
if [ $# -gt 2 ]; then
    shift 2  # Remove config and model_path from args
    EVAL_CMD="${EVAL_CMD} $*"
fi

echo "Running: ${EVAL_CMD}"
echo "------------------------------------------------------"

# Run evaluation
${EVAL_CMD} | tee /s/ls4/users/aamore/rl_a3c_slurm/logs/eval."${SLURM_JOBID}".log

