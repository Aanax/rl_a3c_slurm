"""
The test process: plays with the latest shared model, greedily, and logs the score.

After every finished game (all lives lost) it
  - writes one log line (reward, length, mean reward, frames_total)
  - saves a "latest" checkpoint, and a "best" one when the score is a new record
  - sometimes saves a gif of the game
  - waits 60 seconds before the next game

The option is still sampled here (the model stays in training mode);
only the game action is greedy.
"""
from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import save_checkpoint, setup_logger
import model
from player_util import Agent
import time
import logging
import imageio


def test(args, shared_model, env_conf, frames_total):
    ptitle("Test Agent")
    log_dir_path = f"{args.log_dir}{args.experiment_name}/"
    os.makedirs(log_dir_path, exist_ok=True)
    log_name = f"{args.experiment_name}_{args.parallel_id}_{args.env}_log"
    setup_logger(log_name, rf"{log_dir_path}{args.env}_log_{args.parallel_id}")
    log = logging.getLogger(log_name)
    for key, value in vars(args).items():
        log.info(f"{key}: {value}")

    torch.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    player = Agent(None, env, args, None)
    player.model = model.A3CRules2378OracleTwoLevelMemory(
        player.env.observation_space.shape[0], player.env.action_space, args
    )
    player.state = torch.from_numpy(player.env.reset()).float()

    reward_sum = 0
    reward_total_sum = 0
    num_tests = 0
    max_score = 0
    start_time = time.time()
    rgb_frames = []
    step_counter = 0
    prev_video_at = 0
    if args.gif_image_save_frequency <= 0:
        prev_video_at = float("inf")

    try:
        while True:
            # Take the latest shared weights at the start of every life
            if player.done:
                player.model.load_state_dict(shared_model.state_dict())

            player.action_test()
            step_counter += 1
            frames_total.value += 1
            if frames_total.value > args.total_steps_stop:
                break
            rgb_frames.append(player.env.render(mode='rgb_array'))
            reward_sum += player.reward

            if player.done and not player.env.was_real_done:
                # A life is lost but the game goes on
                player.state = torch.from_numpy(player.env.reset()).float()

            elif player.done and player.env.was_real_done:
                # The game is over
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                log.info(
                    f'Time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}, episode reward {reward_sum}, episode length {player.eps_len}, reward mean {reward_mean:.4f}, frames_total {frames_total.value}'
                )

                if (step_counter - prev_video_at) > args.gif_image_save_frequency:
                    images_dir = f"./gifs/{args.experiment_name}/"
                    os.makedirs(images_dir, exist_ok=True)
                    address = f"{images_dir}{args.env}_run{step_counter}_rew{reward_sum}_parallel_{args.parallel_id}.gif"
                    imageio.mimsave(address, rgb_frames, fps=5)
                    prev_video_at = step_counter
                rgb_frames = []

                weights = player.model.state_dict()
                latest_path = save_checkpoint(weights, args, 'latest', frames_total.value, score=reward_sum)
                print(f'Saved latest checkpoint: {latest_path}')
                if reward_sum > max_score:
                    max_score = reward_sum
                    best_path = save_checkpoint(weights, args, 'best', frames_total.value, score=reward_sum)
                    print(f'Saved best checkpoint: {best_path}')

                reward_sum = 0
                player.eps_len = 0
                state = player.env.reset()
                time.sleep(60)
                player.state = torch.from_numpy(state).float()

    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print("test agent process finished")
