"""
eval.py — play a saved model greedily and save what it did, for draw_eval_gifs.py.

The model is put in eval mode: both the option and the game action are the most
likely ones. An episode ends when a life is lost.

Usage (local):
    python src/eval.py \\
        --config configs/run_config_oracle_two_level_memory_h8_h64_n2.ini \\
        --model-path logs/<experiment>/models/p0/best_steps..._score....dat \\
        --output-dir ./eval_output \\
        --num-episodes 1

Usage (cluster - outputs to logs/<experiment_name>/Eval_.../):
    python src/eval.py --config <config> --model-path <weights> --on-cluster

Output files (.npy, N = number of steps):
    Frames_normalized_orig  frames the model saw        (N, 1, 80, 80)
    ss                      level-1 features s1          (N, 1, 64, 4, 4)
    Q11s                    level-1 action logits        (N, num_actions)
    Q22s                    level-2 option logits        (N, num_options)
    aas                     game action played           (N, 1)
    oos                     option played                (N, 1)
    rewards                 rewards                      (N,)
    Vs, Vs2                 external critics, level 1/2  (N, 1)
    Vs_intrinsic, Vs_intrinsic2
                            intrinsic critics, level 1/2 (N, 1)
"""

from __future__ import print_function, division

import os
import sys
import argparse
import configparser
import json
import contextlib
import io
import datetime

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

import model as model_module
from environment import atari_env


def parse_args():
    """Read the command line and the .ini run config."""
    p = argparse.ArgumentParser(description='Evaluate a two-level oracle model with memory')
    p.add_argument('--config', required=True, help='Path to the .ini run config')
    p.add_argument('--model-path', required=True, help='Path to the .dat weights')
    p.add_argument('--num-episodes', type=int, default=1, help='Number of episodes (default: 1)')
    p.add_argument('--output-dir', default=None, help='Output directory (default: auto-generated)')
    p.add_argument('--on-cluster', action='store_true', help='Save to logs/<experiment_name>/')
    p.add_argument('--max-episode-length', type=int, default=10000, help='Max steps per episode')
    p.add_argument('--render', action='store_true', help='Also save an mp4 of the game')
    p.add_argument('--render-freq', type=int, default=1, help='Render every N-th episode')
    cli = p.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(cli.config)
    c = cfg['DEFAULT']

    args = argparse.Namespace()
    args.env = c.get('env')
    args.env_config = c.get('env_config')
    args.experiment_name = c.get('experiment_name')
    args.num_options = c.getint('num_options')
    args.skip_rate = c.getint('skip_rate')
    args.normalization_alpha = c.getfloat('normalization_alpha')
    args.max_episode_length = cli.max_episode_length

    args.model_path = cli.model_path
    args.num_episodes = cli.num_episodes
    args.output_dir = cli.output_dir
    args.on_cluster = cli.on_cluster
    args.render = cli.render
    args.render_freq = cli.render_freq
    return args


def load_model_and_env(args):
    """Create the game and the model, and load the weights."""
    with open(args.env_config) as f:
        env_conf = json.load(f)
    env_key = args.env.split('-')[0].replace('NoFrameskip', '').replace('Deterministic', '')
    conf = env_conf.get(env_key, env_conf['Default'])

    with contextlib.redirect_stdout(io.StringIO()):  # hide atari_env's prints
        env = atari_env(args.env, conf, args)

    net = model_module.A3CRules2378OracleTwoLevelMemory(
        env.observation_space.shape[0], env.action_space, args
    )
    net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    net.eval()

    print(f"[eval] Env         : {args.env}")
    print(f"[eval] Actions     : {env.action_space.n}")
    print(f"[eval] Options     : {args.num_options}")
    print(f"[eval] Weights     : {args.model_path}")
    return net, env


def run_evaluation(net, env, args):
    """Play the episodes and collect everything the drawing tools need."""
    all_episodes_data = []

    for episode_idx in range(args.num_episodes):
        print(f"\n[eval] Episode {episode_idx + 1}/{args.num_episodes}")
        obs = env.reset()
        previous_s1 = None   # memory starts empty

        record = {
            'Frames_normalized_orig': [], 'ss': [], 'Q11s': [], 'Q22s': [],
            'aas': [], 'oos': [], 'rewards': [],
            'Vs': [], 'Vs2': [], 'Vs_intrinsic': [], 'Vs_intrinsic2': [],
        }
        frames_render = []
        reward_sum = 0
        step_count = 0
        done = False

        while not done and step_count < args.max_episode_length:
            with torch.no_grad():
                out = net(torch.FloatTensor(obs).unsqueeze(0), previous_s1)
            previous_s1 = out['s1']
            action = out['action_logits'].argmax(dim=1).item()

            record['Frames_normalized_orig'].append(obs.copy())
            record['ss'].append(out['s1'].numpy())
            record['Q11s'].append(out['action_logits'].numpy()[0])
            record['Q22s'].append(out['option_logits'].numpy()[0])
            record['aas'].append([action])
            record['oos'].append([out['option'].item()])
            record['Vs'].append(out['value'].numpy()[0])
            record['Vs2'].append(out['value2'].numpy()[0])
            record['Vs_intrinsic'].append(out['value_intrinsic'].numpy()[0])
            record['Vs_intrinsic2'].append(out['value_intrinsic2'].numpy()[0])

            if args.render and episode_idx % args.render_freq == 0:
                render_frame = env.render(mode='rgb_array')
                if render_frame is not None:
                    frames_render.append(render_frame)

            obs, reward, done, info = env.step(action)
            record['rewards'].append(reward)
            reward_sum += reward
            step_count += 1

        print(f"[eval] Episode complete: {step_count} steps, reward = {reward_sum:.2f}")
        episode_data = {key: np.array(values) for key, values in record.items()}
        if frames_render:
            episode_data['frames_render'] = frames_render
        all_episodes_data.append(episode_data)

    return all_episodes_data


def save_eval_results(all_episodes_data, args):
    """Save one .npy per array. Several episodes go to episode_<i>/ subfolders."""
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        model_name = os.path.basename(args.model_path).split('.')[0]
        if args.on_cluster:
            args.output_dir = f"logs/{args.experiment_name}/Eval_{timestamp}_{model_name}/"
        else:
            args.output_dir = f"./Eval_{timestamp}_{model_name}/"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[eval] Saving results to: {args.output_dir}")

    model_prefix = os.path.basename(args.model_path).replace('.dat', '').replace('.pth', '')
    for episode_idx, episode_data in enumerate(all_episodes_data):
        if len(all_episodes_data) > 1:
            episode_dir = os.path.join(args.output_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)
        else:
            episode_dir = args.output_dir

        for key, data in episode_data.items():
            if key == 'frames_render':
                try:
                    import imageio
                    render_path = os.path.join(episode_dir, f"{model_prefix}_render.mp4")
                    imageio.mimsave(render_path, data, fps=30)
                    print(f"[eval] Saved render: {render_path}")
                except Exception as e:
                    print(f"[eval] Warning: Could not save render: {e}")
            else:
                filepath = os.path.join(episode_dir, f"{model_prefix}_{key}.npy")
                np.save(filepath, data)
                print(f"[eval] Saved {key}: {filepath} (shape: {data.shape})")

    print(f"\n[eval] All results saved to: {args.output_dir}")
    return args.output_dir


def main():
    args = parse_args()
    net, env = load_model_and_env(args)
    all_episodes_data = run_evaluation(net, env, args)
    save_eval_results(all_episodes_data, args)
    env.close()


if __name__ == '__main__':
    main()
