from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging
import imageio
import copy


def test(args, shared_model, env_conf, frames_total):
    ptitle("Test Agent")
    gpu_id = args.gpu_ids[-1]
    log_dir_path = f"{args.log_dir}{args.experiment_name}/"
    os.makedirs(log_dir_path, exist_ok=True)
    log_file_path = rf"{log_dir_path}{args.env}_log_{args.parallel_id}"
    setup_logger(f"{args.experiment_name}_{args.parallel_id}_{args.env}_log", log_file_path)
    log = logging.getLogger(f"{args.experiment_name}_{args.parallel_id}_{args.env}_log")
    d_args = vars(args)
    for k in d_args.keys():
        log.info(f"{k}: {d_args[k]}")

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0], player.env.action_space, args)

    if args.tensorboard_logger:
        from torch.utils.tensorboard import SummaryWriter
        dummy_input = (torch.zeros(1, player.env.observation_space.shape[0], 80, 80), torch.zeros(1, args.hidden_size), torch.zeros(1, args.hidden_size),  )
        writer_path = f"runs/{args.experiment_name}/{args.experiment_name}_{args.env}_training_{args.parallel_id}"
        writer = SummaryWriter(writer_path)
        writer.add_graph(player.model, dummy_input, False)
        writer.close()

    player.state = player.env.reset()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = torch.from_numpy(player.state).float().cuda()
    else:
        player.state = torch.from_numpy(player.state).float()

    flag = True
    max_score = 0
    rgb_frames = []
    step_counter = 0
    prev_video_at = 0
    if args.gif_image_save_frequency <= 0:
        prev_video_at = float("inf")
        
    try:
        while 1:
            if player.done:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.model.load_state_dict(shared_model.state_dict())
                else:
                    player.model.load_state_dict(shared_model.state_dict())

            player.action_test()
            step_counter += 1
            rgb_frames.append(player.env.render(mode='rgb_array'))
            reward_sum += player.reward

            if player.done and not player.env.was_real_done:
                state = player.env.reset()
                player.state = torch.from_numpy(state).float()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = player.state.cuda()
            elif player.done and player.env.was_real_done:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                log.info(
                    f'Time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))}, episode reward {reward_sum}, episode length {player.eps_len}, reward mean {reward_mean:.4f}, frames_total {frames_total.value}'
                )
                if args.tensorboard_logger:
                    writer.add_scalar(
                        f"{args.env}_Episode_Rewards", reward_sum, num_tests
                    )
                    for name, weight in player.model.named_parameters():
                        writer.add_histogram(name, weight, num_tests)
                if (step_counter - prev_video_at) > args.gif_image_save_frequency:
                    images_dir = f"./gifs/{args.experiment_name}/"
                    os.makedirs(images_dir, exist_ok=True)
                    address = f"{images_dir}{args.env}_run{step_counter}_rew{reward_sum}_parallel_{args.parallel_id}.gif"
                    saveanimation(rgb_frames, address=address)
                    prev_video_at = step_counter
                rgb_frames = []
                if (args.save_max and reward_sum >= max_score) or not args.save_max:
                    if reward_sum >= max_score:
                        max_score = reward_sum
                    if gpu_id >= 0:
                        with torch.cuda.device(gpu_id):
                            state_to_save = player.model.state_dict()
                            torch.save(
                                state_to_save, f"{args.save_model_dir}{args.env}.dat"
                            )
                    else:
                        state_to_save = player.model.state_dict()
                        torch.save(
                            state_to_save, f"{args.save_model_dir}{args.env}.dat"
                        )

                reward_sum = 0
                player.eps_len = 0
                state = player.env.reset()
                time.sleep(60)
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = torch.from_numpy(state).float().cuda()
                else:
                    player.state = torch.from_numpy(state).float()

    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print("test agent process finished")
        if args.tensorboard_logger:
            writer.close()


def saveanimation(frames, address="./demo/movie_base.gif"):
    """
    This method ,given the frames of images make the gif and save it in the folder

    params:
        frames:method takes in the array or np.array of images
        address:(optional)given the address/location saves the gif on that location
                otherwise save it to default address './demo/movie_base.gif'

    return :
        none
    """
    imageio.mimsave(address, frames, fps=5)
