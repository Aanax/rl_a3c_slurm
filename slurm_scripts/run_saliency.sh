#!/bin/sh
#SBATCH --job-name=saliency_eval
#SBATCH -D /s/ls4/users/aamore/rl_a3c_pytorch/
#SBATCH -o /s/ls4/users/aamore/rl_a3c_pytorch/logs/%j.out
#SBATCH -e /s/ls4/users/aamore/rl_a3c_pytorch/logs/%j.err
#SBATCH -t 02:00:00
#SBATCH --nodes 1
#SBATCH --cpus-per-task=2
#SBATCH -p hpc4-el7-gpu-3d

# Usage:
#   sbatch slurm_scripts/run_saliency.sh <config> <model_path> [num_frames]
#
# Arguments:
#   $1  path to .ini run config  (e.g. configs/run_config_our5.ini)
#   $2  path to .pth checkpoint  (e.g. trained_models/best_model.pth)
#   $3  number of frames to eval (optional, default: 200)
#
# Output:
#   Per-frame PNGs + saliency.gif written to saliency_output/<job_id>/
#   SLURM stdout/stderr in logs/%j.out / logs/%j.err
#   Tee'd human-readable log in logs/saliency.<job_id>.log

CONFIG=${1}
MODEL_PATH=${2}
NUM_FRAMES=${3:-200}

# ── conda environment setup (mirrors run_eval.sh) ─────────────────────────────
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
echo "NUM_FRAMES : ${NUM_FRAMES}"
echo "------------------------------------------------------"

# ── run saliency eval ─────────────────────────────────────────────────────────
python /s/ls4/users/aamore/rl_a3c_pytorch/src/eval_saliency.py \
    --config     "${CONFIG}" \
    --model-path "${MODEL_PATH}" \
    --num-frames "${NUM_FRAMES}" \
    --output-dir "/s/ls4/users/aamore/rl_a3c_pytorch/saliency_output/${SLURM_JOBID}" \
    | tee /s/ls4/users/aamore/rl_a3c_pytorch/logs/saliency."${SLURM_JOBID}".log
