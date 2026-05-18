"""
eval.py — Evaluation script for Hierarchial models (Hierarchial, Hierarchial_memory, Hierarchial_memory_memrelu).

Runs a greedy policy rollout, saves level1 logits, level2 logits, actions, rewards, 
values1, values2, and normalized frames in a format compatible with the drawing tools.

Usage (local):
    python src/eval.py \\
        --config configs/run_config_our.ini \\
        --model-path trained_models/PongNoFrameskip-v4.dat \\
        --output-dir ./eval_output \\
        --num-episodes 1

Usage (cluster - outputs to logs/ folder):
    python src/eval.py \\
        --config configs/run_config_our.ini \\
        --model-path trained_models/PongNoFrameskip-v4.dat \\
        --num-episodes 1 \\
        --on-cluster

Output format (saved as .npy files):
    Frames_normalized_orig.npy - Preprocessed frames (N, C, H, W)
    Q11s.npy - Level 1 logits (N, num_actions)
    Q22s.npy - Level 2 logits (N, 16) 
    aas.npy - Selected actions (N, 1)
    rewards.npy - Rewards (N,)
    Vs.npy - Level 1 values (N, 1)
    Vs2.npy - Level 2 values (N, 1)
    ss.npy - Level 1 encoder features (N, 64, 4, 4)

    sbatch slurm_scripts/run_eval.sh ./configs/run_config_our3.ini ./logs/Hierarchial_a2a1_connect_32w_g_09_g2_099_15.dat --zero-a2
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
import torch.nn.functional as F

# Path setup
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

import model as model_module
from environment import atari_env
from player_util import Agent


def parse_args():
    """Parse CLI flags and merge with the .ini run-config."""
    p = argparse.ArgumentParser(
        description='Evaluation script for Hierarchial models'
    )
    p.add_argument('--config', required=True,
                   help='Path to .ini run config (e.g. configs/run_config_our.ini)')
    p.add_argument('--model-path', required=True,
                   help='Path to .dat/.pth weights checkpoint')
    p.add_argument('--num-episodes', type=int, default=1,
                   help='Number of episodes to evaluate (default: 1)')
    p.add_argument('--output-dir', default=None,
                   help='Output directory for eval files (default: auto-generated)')
    p.add_argument('--on-cluster', action='store_true',
                   help='Running on cluster - save to logs/ folder with proper naming')
    p.add_argument('--gpu-id', type=int, default=-1,
                   help='GPU to use [-1 CPU only] (default: -1)')
    p.add_argument('--max-episode-length', type=int, default=10000,
                   help='Maximum episode length (default: 10000)')
    p.add_argument('--render', action='store_true',
                   help='Render gameplay (default: False)')
    p.add_argument('--render-freq', type=int, default=1,
                   help='Frequency to render (default: 1)')
    p.add_argument('--zero-a2', action='store_true',
                   help='Append _zeroing suffix to model_type (zeros a2 input to actor)')
    
    cli = p.parse_args()

    # Read .ini config
    cfg = configparser.ConfigParser()
    cfg.read(cli.config)

    args = argparse.Namespace()

    # Model and environment parameters from config
    args.env = cfg.get('DEFAULT', 'env', fallback='PongNoFrameskip-v4')
    args.hidden_size = cfg.getint('DEFAULT', 'hidden_size', fallback=1024)
    args.gamma = cfg.getfloat('DEFAULT', 'gamma', fallback=0.9)
    args.gamma_memory = cfg.getfloat('DEFAULT', 'gamma_memory', fallback=0.9)
    args.gamma2 = cfg.getfloat('DEFAULT', 'gamma2', fallback=0.99)
    args.tau = cfg.getfloat('DEFAULT', 'tau', fallback=1.0)
    args.skip_rate = cfg.getint('DEFAULT', 'skip_rate', fallback=4)
    args.max_episode_length = cli.max_episode_length
    args.input_normalization_class = cfg.get('DEFAULT', 'input_normalization_class', fallback='NormalizedEnv')
    args.model_type = cfg.get('DEFAULT', 'model_type', fallback='Hierarchial')
    args.env_config = cfg.get('DEFAULT', 'env_config', fallback='configs/envs_config.json')
    args.normalization_alpha = cfg.getfloat('DEFAULT', 'normalization_alpha', fallback=0.9999)
    args.monitor_s = False
    args.use_rmsnorm = False
    
    gpu_str = cfg.get('DEFAULT', 'gpu_ids', fallback='-1')
    cfg_gpu_ids = [int(x.strip()) for x in gpu_str.split(',') if x.strip()]
    args.gpu_ids = [cli.gpu_id] if cli.gpu_id >= 0 else cfg_gpu_ids

    # Eval-specific settings
    args.model_path = cli.model_path
    args.num_episodes = cli.num_episodes
    args.output_dir = cli.output_dir
    args.on_cluster = cli.on_cluster
    args.render = cli.render
    args.render_freq = cli.render_freq
    args.gpu_id = cli.gpu_id
    args.zero_a2 = cli.zero_a2

    return args


def load_model_and_env(args):
    """Instantiate the Atari environment and load model weights."""
    env_conf = json.load(open(args.env_config))
    env_key = args.env.split('-')[0].replace('NoFrameskip', '').replace('Deterministic', '')
    conf = env_conf.get(env_key, env_conf['Default'])

    # Suppress atari_env's startup prints
    with contextlib.redirect_stdout(io.StringIO()):
        env = atari_env(args.env, conf, args)

    num_inputs = env.observation_space.shape[0]

    # If --zero-a2 is specified, append '_zeroing' suffix to model type
    if getattr(args, 'zero_a2', False):
        args.model_type = args.model_type + '_zeroing'
    
    model_cls = getattr(model_module, args.model_type)
    net = model_cls(num_inputs, env.action_space, args)
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()

    # Move to GPU if requested
    gpu_id = args.gpu_id
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            net = net.cuda()

    print(f"[eval] Model       : {args.model_type}")
    print(f"[eval] Env         : {args.env} (num_inputs={num_inputs})")
    print(f"[eval] Actions     : {env.action_space.n}")
    print(f"[eval] Weights     : {args.model_path}")
    print(f"[eval] GPU         : {gpu_id if gpu_id >= 0 else 'CPU'}")

    return net, env


def _reset_model_memory(net):
    """Reset model internal memory if applicable."""
    if hasattr(net, 'running_mem'):
        net.running_mem = torch.zeros((1, 64, 4, 4))
    if hasattr(net, 'prev_x_conv'):
        net.prev_x_conv = None


def run_evaluation(net, env, args):
    """
    Run evaluation episode(s) and collect all relevant data.
    
    Returns dictionaries with collected data for each episode.
    """
    gpu_id = args.gpu_id
    
    all_episodes_data = []
    
    for episode_idx in range(args.num_episodes):
        print(f"\n[eval] Episode {episode_idx + 1}/{args.num_episodes}")
        
        # Reset environment and model memory
        obs = env.reset()
        _reset_model_memory(net)
        
        # Initialize tracking lists
        frames_normalized_orig = []
        frames_render = []
        ss = []  # Level 1 encoder features
        Q11s = []  # Level 1 logits
        Q22s = []  # Level 2 logits
        aas = []  # Actions taken
        rewards = []  # Rewards received
        Vs = []  # Level 1 values
        Vs2 = []  # Level 2 values
        
        reward_sum = 0
        step_count = 0
        done = False
        
        while not done and step_count < args.max_episode_length:
            # Prepare observation tensor
            obs_t = torch.FloatTensor(obs).unsqueeze(0)  # (1, C, H, W)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    obs_t = obs_t.cuda()
            
            # Forward pass through model
            with torch.no_grad():
                model_output = net(obs_t, torch.zeros(1), torch.zeros(1))
            
            # Parse model output: (V1, a1, hx, cx, None, None, V2, a2_logits)
            V1 = model_output[0]
            a1_logits = model_output[1]
            V2 = model_output[6]
            a2_logits = model_output[7]
            
            # Get action probabilities and select action (greedy)
            prob = F.softmax(a1_logits, dim=1)
            action = prob.cpu().numpy().argmax(axis=1)[0]
            
            # Store data
            frames_normalized_orig.append(obs.copy())
            ss.append(net.s_values[-1].cpu().numpy() if hasattr(net, 's_values') and net.s_values 
                      else torch.zeros(1, 64, 4, 4).numpy())  # Level 1 features
            Q11s.append(a1_logits.cpu().numpy()[0])  # Level 1 logits
            Q22s.append(a2_logits.cpu().numpy()[0])  # Level 2 logits (16-dim)
            aas.append([action])  # Action taken
            Vs.append(V1.cpu().numpy()[0])  # Level 1 value
            Vs2.append(V2.cpu().numpy()[0])  # Level 2 value
            
            # Render if requested
            if args.render and episode_idx % args.render_freq == 0:
                render_frame = env.render(mode='rgb_array')
                if render_frame is not None:
                    frames_render.append(render_frame)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            rewards.append(reward)
            step_count += 1
            
            # Clear s_values after storing to prevent memory buildup
            if hasattr(net, 's_values'):
                net.s_values = []
        
        print(f"[eval] Episode complete: {step_count} steps, reward = {reward_sum:.2f}")
        
        # Package episode data
        episode_data = {
            'Frames_normalized_orig': np.array(frames_normalized_orig),
            'ss': np.array(ss),
            'Q11s': np.array(Q11s),
            'Q22s': np.array(Q22s),
            'aas': np.array(aas),
            'rewards': np.array(rewards),
            'Vs': np.array(Vs),
            'Vs2': np.array(Vs2),
        }
        
        if frames_render:
            episode_data['frames_render'] = frames_render
            
        all_episodes_data.append(episode_data)
    
    return all_episodes_data


def save_eval_results(all_episodes_data, args):
    """
    Save evaluation results to disk in the format expected by drawing tools.
    
    For single episode: saves directly to output folder
    For multiple episodes: creates subfolders
    """
    # Generate output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        model_name = os.path.basename(args.model_path).split('.')[0]
        
        zeroing_suffix = "_zeroing" if getattr(args, 'zero_a2', False) else ""
        
        if args.on_cluster:
            # On cluster: save to logs/experiment_name/Eval_.../
            exp_name = getattr(args, 'experiment_name', 'eval')
            args.output_dir = f"logs/{exp_name}/Eval_{timestamp}_{model_name}{zeroing_suffix}/"
        else:
            args.output_dir = f"./Eval_{timestamp}_{model_name}{zeroing_suffix}/"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[eval] Saving results to: {args.output_dir}")
    
    for episode_idx, episode_data in enumerate(all_episodes_data):
        # For multiple episodes, create subfolders
        if len(all_episodes_data) > 1:
            episode_dir = os.path.join(args.output_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)
        else:
            episode_dir = args.output_dir
        
        # Get model name for file prefixes (strip path and extension)
        model_prefix = os.path.basename(args.model_path).replace('.dat', '').replace('.pth', '')
        
        # Save all arrays with the naming convention from old eval
        for key, data in episode_data.items():
            if key == 'frames_render':
                # Save render frames as mp4 if available
                try:
                    import imageio
                    render_path = os.path.join(episode_dir, f"{model_prefix}_render.mp4")
                    imageio.mimsave(render_path, data, fps=30)
                    print(f"[eval] Saved render: {render_path}")
                except Exception as e:
                    print(f"[eval] Warning: Could not save render: {e}")
            else:
                # Save numpy arrays
                filepath = os.path.join(episode_dir, f"{model_prefix}_{key}.npy")
                np.save(filepath, data)
                print(f"[eval] Saved {key}: {filepath} (shape: {data.shape})")
    
    print(f"\n[eval] All results saved to: {args.output_dir}")
    return args.output_dir


def main():
    args = parse_args()
    
    # Load model and environment
    net, env = load_model_and_env(args)
    
    # Run evaluation
    all_episodes_data = run_evaluation(net, env, args)
    
    # Save results
    output_dir = save_eval_results(all_episodes_data, args)
    
    # Print summary
    print(f"\n[eval] Evaluation complete!")
    print(f"[eval] Output directory: {output_dir}")
    
    env.close()


if __name__ == '__main__':
    main()
