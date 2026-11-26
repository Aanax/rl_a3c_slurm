# RL A3C Pytorch: two-level oracle agent with memory

A3C (Asynchronous Advantage Actor-Critic) on Atari games, CPU only.

The agent has two levels:

- **Level 2** chooses an *option* (a number `0 … num_options-1`) every step.
- **Level 1** chooses the *game action* every step. It sees the screen features `s1`,
  a **memory** (how `s1` changed since the previous step), and the current option.

Each level has an actor, an external critic (game reward), an intrinsic critic (oracle reward)
and an oracle that guesses how its features will change. Both actors learn from their level's
intrinsic advantage. See the top of `src/train.py` for what each part learns from.

## Project Structure

- `src/`
  - `main.py` - reads the config, starts the test process and the training workers
  - `train.py` - one training worker: plays a batch, computes the losses, updates the shared model
  - `test.py` - plays greedily with the latest weights, logs scores, saves checkpoints
  - `model.py` - the network (`A3CRules2378OracleTwoLevelMemory`)
  - `player_util.py` - the `Agent`: plays steps and stores what training needs
  - `environment.py` - Atari wrappers (80x80 grayscale, normalized, frame skip, life = episode)
  - `shared_optim.py` - Adam with shared state for all workers
  - `utils.py` - logging, config reading, checkpoints
  - `eval.py` - plays a saved model and saves arrays for `draw_eval_gifs.py`
  - `draw_eval_gifs.py`, `plot_train_curves.py` - plots and videos
- `configs/`
  - `run_config_oracle_two_level_memory_h8_h64_n2.ini` - Pong, horizons 8 / 64, 2 options
  - `run_config_oracle_two_level_memory_pacman.ini` - MsPacman, horizons 4 / 16, 8 options
  - `envs_config.json` - per-game frame cropping
- `slurm_scripts/` - cluster job scripts
- `tests/` - `python -m unittest discover tests`

## Launching

Locally:
```bash
python src/main.py configs/run_config_oracle_two_level_memory_h8_h64_n2.ini 0
```
The second argument is the `parallel_id` that tells several runs of one config apart.

On the cluster, 5 runs:
```bash
bash slurm_scripts/our/parallel_oracle_two_level_memory_h8_h64_n2.sh 5 > job_ids.txt
python slurm_scripts/batch_cancel.py job_ids.txt   # to cancel them
```

## Configuration

| key | meaning |
| --- | --- |
| `gamma`, `gamma2` | discounts of level 1 and level 2 (horizon = 1 / (1 - gamma)) |
| `gamma_oracle`, `gamma_oracle2` | discounts of the oracle targets of level 1 and level 2 |
| `num_options` | number of options level 2 chooses from |
| `w_restoration_loss` | weight of the oracle losses |
| `num_steps` | steps per training batch |
| `workers` | number of training processes |
| `total_steps_stop` | stop after this many frames |

## Output

Everything goes to `logs/{experiment_name}/`:
- `{env}_log_{parallel_id}` - one line per finished game (reward, length, mean reward, frames)
- `models/p{parallel_id}/` - `latest_*.dat` and `best_*.dat` checkpoints
- `losses_rank{rank}.csv`, `cosine_const_rank{rank}.csv` - per-batch losses and oracle cosines
- `s_monitor_*.pkl` - `s1` of every step of some games

Game GIFs go to `gifs/{experiment_name}/`.

## References

Based on Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)
