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
    Q22s.npy - Level 2 logits (N, num_options)
    aas.npy - Selected actions (N, 1)
    oos.npy - Option indices actually played (N, 1)
    beta2s.npy - Level-2 termination coeff for chosen option (N, 1)
    terminated1s.npy - Level-1 always resamples; 1 every step (N, 1)
    terminated2s.npy - Level-2 terminate samples 0/1 (N, 1)
    betas.npy / beta_logits.npy / beta_active.npy / beta_samples.npy - legacy beta dumps
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
                   help='Use _zeroing variant (zero a21 interactor; a1+a2 play normally)')
    p.add_argument('--zero-a1', action='store_true',
                   help='Use _zeroing2 variant (zero a1 actor; a21+a2 play normally)')
    cli = p.parse_args()

    # Read .ini config
    cfg = configparser.ConfigParser()
    cfg.read(cli.config)

    args = argparse.Namespace()

    # Model and environment parameters from config
    args.env = cfg.get('DEFAULT', 'env', fallback='PongNoFrameskip-v4')
    args.hidden_size = cfg.getint('DEFAULT', 'hidden_size', fallback=1024)
    args.gamma = cfg.getfloat('DEFAULT', 'gamma', fallback=0.0)
    print("Using gamma ", args.gamma)
    args.gamma_memory = cfg.getfloat('DEFAULT', 'gamma_memory', fallback=0.0)
    print("Using gamma_memory ", args.gamma_memory)
    args.gamma2 = cfg.getfloat('DEFAULT', 'gamma2', fallback=0.99)
    print("Using gamma2 ", args.gamma2)
    args.gamma_actor = cfg.getfloat('DEFAULT', 'gamma_actor', fallback=args.gamma)
    args.gamma2_actor = cfg.getfloat(
        'DEFAULT', 'gamma2_actor', fallback=args.gamma2
    )
    print("Using gamma_actor ", args.gamma_actor)
    print("Using gamma2_actor ", args.gamma2_actor)
    args.tau = cfg.getfloat('DEFAULT', 'tau', fallback=1.0)
    args.skip_rate = cfg.getint('DEFAULT', 'skip_rate', fallback=4)
    args.max_episode_length = cli.max_episode_length
    args.input_normalization_class = cfg.get('DEFAULT', 'input_normalization_class', fallback='NormalizedEnv')
    args.model_type = cfg.get('DEFAULT', 'model_type', fallback='Hierarchial_interactor_options')
    args.num_options = cfg.getint('DEFAULT', 'num_options', fallback=8)
    args.env_config = cfg.get('DEFAULT', 'env_config', fallback='configs/envs_config.json')
    args.normalization_alpha = cfg.getfloat('DEFAULT', 'normalization_alpha', fallback=0.9999)
    args.monitor_s = False
    args.use_rmsnorm = False
    args.use_beta_termination = cfg.getboolean(
        'DEFAULT', 'use_beta_termination', fallback=False
    )
    
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
    args.zero_a1 = cli.zero_a1
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

    if args.zero_a1 and args.zero_a2:
        raise ValueError('Use only one of --zero-a1 or --zero-a2, not both')
    if args.zero_a1:
        args.model_type = args.model_type + '_zeroing2'
    elif args.zero_a2:
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
    """Reset sticky action / option state between episodes."""
    if hasattr(net, 'reset_persistent_actions'):
        net.reset_persistent_actions()
    elif hasattr(net, 'current_option'):
        net.current_option = None
    if hasattr(net, 'beta_values'):
        net.beta_values = []
    if hasattr(net, 'beta_logits_values'):
        net.beta_logits_values = []
    if hasattr(net, 'last_beta_logits'):
        net.last_beta_logits = None


def _model_uses_legacy_beta(net, args):
    """Old models that dump per-option beta vectors via monitor_beta."""
    return args.use_beta_termination and hasattr(net, 'beta_linear')


def _to_scalar_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(x)


def _to_numpy_1d(x):
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr).reshape(-1)


def _parse_model_output(model_output):
    """Parse forward() return for option models into a uniform dict.

    Prefers HierarchialLevelsOutput named fields when available.
    Legacy models fall back to positional layout:
      concat (11):  [0]=V1 [1]=a1_logits [6]=V2 [7]=a2_logits
                    [8]=a2 [9]=beta2 [10]=terminated2
      interactor (14): [10]=a2 [11]=beta [13]=terminated2
    """
    from model import HierarchialLevelsOutput

    if isinstance(model_output, HierarchialLevelsOutput):
        return {
            'V1': model_output.V1,
            'action_logits': model_output.a1_logits,
            'V2': model_output.V2,
            'a2_logits': model_output.a2_logits,
            'a1_sample': model_output.a1,
            'a2_sample': model_output.a2,
            'beta2': model_output.beta2,
            'terminated1': model_output.terminated1,
            'terminated2': model_output.terminated2,
            'has_levels_betas': True,
        }

    if len(model_output) <= 11:
        a2_sample = model_output[8]
        beta_active = model_output[9]
        option_terminated = model_output[10] if len(model_output) > 10 else False
    else:
        a2_sample = model_output[10]
        beta_active = model_output[11]
        option_terminated = model_output[13] if len(model_output) > 13 else False

    return {
        'V1': model_output[0],
        'action_logits': model_output[1],
        'V2': model_output[6],
        'a2_logits': model_output[7],
        'a1_sample': None,
        'a2_sample': a2_sample,
        'beta2': beta_active,
        'terminated1': None,
        'terminated2': option_terminated,
        'has_levels_betas': False,
    }


def run_evaluation(net, env, args):
    """
    Run evaluation episode(s) and collect all relevant data.
    
    Returns dictionaries with collected data for each episode.
    """
    gpu_id = args.gpu_id
    log_legacy_beta = _model_uses_legacy_beta(net, args)
    if log_legacy_beta:
        net.monitor_beta = True
        net.beta_values = []
        print("[eval] Legacy beta logging enabled")
    
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
        betas = []  # Per-option termination betas (legacy)
        beta_logits = []  # Pre-sigmoid beta logits (legacy)
        beta_active = []  # Active termination beta for current option (legacy)
        oos = []  # Options actually played
        beta_samples = []  # Bernoulli terminate samples (legacy alias of terminated2)
        beta2s = []  # Level-2 termination coeff for chosen option
        terminated1s = []  # Level-1 always resamples
        terminated2s = []  # Level-2 terminate samples
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
            
            parsed = _parse_model_output(model_output)
            V1 = parsed['V1']
            a1_logits = parsed['action_logits']
            V2 = parsed['V2']
            a2_logits = parsed['a2_logits']
            a1_sample = parsed['a1_sample']
            a2_sample = parsed['a2_sample']
            beta2_step = parsed['beta2']
            terminated1 = parsed['terminated1']
            terminated2 = parsed['terminated2']
            
            # Prefer sampled a1 from Hierarchial_levels; otherwise greedy argmax
            if a1_sample is not None:
                action = int(_to_numpy_1d(a1_sample)[0])
            else:
                prob = F.softmax(a1_logits, dim=1)
                action = int(prob.cpu().numpy().argmax(axis=1)[0])
            
            # Store data
            frames_normalized_orig.append(obs.copy())
            ss.append(net.s_values[-1].cpu().numpy() if hasattr(net, 's_values') and net.s_values 
                      else torch.zeros(1, 64, 4, 4).numpy())  # Level 1 features
            Q11s.append(a1_logits.cpu().numpy()[0])  # Level 1 logits
            Q22s.append(a2_logits.cpu().numpy()[0])  # Level 2 logits
            aas.append([action])  # Action taken
            oos.append([int(_to_numpy_1d(a2_sample)[0])])  # Option actually played

            if beta2_step is not None:
                beta2s.append([_to_scalar_float(beta2_step)])
            if terminated1 is not None:
                terminated1s.append([_to_scalar_float(terminated1)])
            if terminated2 is not None:
                terminated2s.append([_to_scalar_float(terminated2)])

            if log_legacy_beta:
                betas.append(net.beta_values[-1].numpy()[0])
                beta_logits.append(net.last_beta_logits.cpu().numpy()[0])
                if beta2_step is not None:
                    beta_active.append(_to_numpy_1d(beta2_step))
                beta_samples.append([_to_scalar_float(terminated2)])

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
            
            # Clear cached forward-pass values after storing
            if hasattr(net, 's_values'):
                net.s_values = []
            if log_legacy_beta and hasattr(net, 'beta_values'):
                net.beta_values = []
            if log_legacy_beta and hasattr(net, 'beta_logits_values'):
                net.beta_logits_values = []
        
        print(f"[eval] Episode complete: {step_count} steps, reward = {reward_sum:.2f}")
        
        # Package episode data
        episode_data = {
            'Frames_normalized_orig': np.array(frames_normalized_orig),
            'ss': np.array(ss),
            'Q11s': np.array(Q11s),
            'Q22s': np.array(Q22s),
            'aas': np.array(aas),
            'oos': np.array(oos),
            'rewards': np.array(rewards),
            'Vs': np.array(Vs),
            'Vs2': np.array(Vs2),
        }
        if beta2s:
            episode_data['beta2s'] = np.array(beta2s)
        if terminated1s:
            episode_data['terminated1s'] = np.array(terminated1s)
        if terminated2s:
            episode_data['terminated2s'] = np.array(terminated2s)
        if log_legacy_beta:
            episode_data['betas'] = np.array(betas)
            episode_data['beta_logits'] = np.array(beta_logits)
            episode_data['beta_active'] = np.array(beta_active)
            episode_data['beta_samples'] = np.array(beta_samples)
        
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

        if args.zero_a1:
            zeroing_suffix = '_zeroing2'
        elif args.zero_a2:
            zeroing_suffix = '_zeroing'
        else:
            zeroing_suffix = ''

        if args.on_cluster:
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
