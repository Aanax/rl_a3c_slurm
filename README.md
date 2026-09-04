# RL A3C Pytorch

This project implements Asynchronous Advantage Actor-Critic (A3C) reinforcement learning algorithm in PyTorch for training agents on Atari games.

## Project Structure

- `src/` - Main source code
  - `main.py` - Main entry point for running experiments
  - `train.py` - Training worker processes
  - `test.py` - Test agent for evaluation (running in parallel wit main training)
  - `model.py` - A3C model architecture
  - `environment.py` - Atari environment wrapper
  - `player_util.py` - Agent utilities (defines Agent class)
  - `shared_optim.py` - Shared optimizer for multiprocessing
  - `utils.py` - Utility functions

- `configs/` - Configuration files
  - `run_config_base.ini` - Base configuration template
  - `envs_config.json` - Environment specific settings (from original repo)

- `logs/` - Experiment logs
  - Organized by experiment name and parallel ID

- `trained_models/` - Saved model checkpoints

- `runs/` - TensorBoard logs (original repo implementation)

- `slurm_scripts/` - SLURM job submission scripts
  - `batch_cancel.py` - Cancel SLURM jobs from a file

- `graphics_utils/` - Plotting utilities

## Launching Experiments

### single SLURM Job Submission

For cluster training, use the SLURM scripts:

1. Create your experiment config:
```bash
cp configs/run_config_base.ini configs/your_experiment.ini
# Edit your_experiment.ini with your parameters
```

2. Create your experiment .sh (TODO make one runner and pass config as param)
```bash
cp slurm_scripts/base/run_config_base.ini slurm_scripts/your_experiment_folder/run_your_experiment.sh
# Edit your_experiment.ini with your parameters
```

3. Submit the SLURM job:

Example:
```bash
sbatch slurm_scripts/base/run_one_run.sh configs/your_experiment.ini my_run 0
```


### Batch run (TODO)

```bash
sbatch parallel.sh 5 > job_ids.txt
```

### Batch Job Cancellation

To cancel jobs listed in a file:
```bash
python slurm_scripts/batch_cancel.py job_ids.txt
```

## Configuration

Main parameters in config file:

- `env`: Atari environment name (e.g., PongNoFrameSkip-v4)
- `workers`: Number of parallel workers
- `lr`: Learning rate
- `gamma`: Discount factor
- `seed`: Random seed
- `save_max`: Whether to save only best models
- `tensorboard_logger`: Enable TensorBoard logging

## Monitoring Progress

- Logs are saved in `logs/{experiment_name}_{parallel_id}/`
- TensorBoard logs in `runs/{experiment_name}_{parallel_id}_{env}_training`
- GIFs and images in `gifs/{experiment_name}_{parallel_id}/`
- Model checkpoints in `trained_models/`

## Output

Training logs include:
- Episode reward, length, and mean reward
- Total frames processed across all workers
- Model saves (if enabled)

## References

Based on Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)




## Useful
drawing gifs
```python src/draw_eval_gifs.py logs/eval/Eval_2026-09-03_19:58:47_best_steps7888809_score1890 --panels beta2 option2 logits2 logits1 --no-download```
