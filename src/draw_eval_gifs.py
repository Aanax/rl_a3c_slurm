"""
draw_eval_gifs.py — Download evaluation results from remote server and create visualization GIFs.

This script downloads evaluation artifacts from a remote server (via SSH/SFTP),
then creates two MP4 videos:
  1. actions.mp4 - Shows logits1, level2 logits, played option, and sampled beta
  2. VS_short.mp4 - Shows V1, V2 values and rewards over time

Usage:
    python src/draw_eval_gifs.py EVAL_FOLDER_NAME [options]
    
Arguments:
    EVAL_FOLDER_NAME - Name of the eval folder on remote server (e.g., Eval_2024-12-04_21:58:10_cosineFix2_try2.468102)
    
Options:
    --local-dir LOCAL_DIR     Local directory to download files (default: same as eval folder name)
    --server SERVER           Server SSH name/address (default: ui4.computing.kiae.ru)
    --username USERNAME       SSH username (default: aamore)
    --remote-path PATH        Remote project path (default: /home/users/aamore/rl_a3c_slurm/)
    --no-download             Skip download (use local files if they exist)
    --fps FPS                 FPS for output GIFs (default: 3)
    --start-idx START         Start frame index (default: 0)
    --stop-idx STOP           Stop frame index (default: 300, -1 for all)
    --window-size SIZE        Window size for value plots (default: 34)

Example:
    python src/draw_eval_gifs.py Eval_2026-03-29_00:21:09_PongNoFrameskip-v4 --remote-path /home/users/aamore/rl_a3c_pytorch/logs/eval/
"""

import torch
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import imageio
from tqdm import tqdm
import os
import subprocess
import sys
import time
import argparse
from matplotlib import gridspec

# Optional paramiko import (only needed for remote downloads)
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    paramiko = None


def _panel_ylim(panel_cfg, val_data):
    if 'ylim' in panel_cfg:
        return panel_cfg['ylim']
    return np.min(val_data), np.max(val_data)


def _plot_panel(ax, window, panel_cfg):
    legend = panel_cfg.get('legend', None)
    plot_style = panel_cfg.get('plot_style', 'line')
    x = np.arange(window.shape[0])
    markersize = panel_cfg.get('markersize', 6)

    if panel_cfg.get('plot_columns', False) and window.ndim == 2 and window.shape[1] > 1:
        for col in range(window.shape[1]):
            label = legend[col] if legend and col < len(legend) else f'{col}'
            if plot_style == 'dots':
                ax.plot(
                    x, window[:, col], 'o', label=label,
                    markersize=markersize, linestyle='None',
                )
            else:
                ax.plot(x, window[:, col], label=label)
    elif window.ndim == 2:
        if plot_style == 'dots':
            for col in range(window.shape[1]):
                ax.plot(x, window[:, col], 'o', markersize=markersize, linestyle='None')
        else:
            ax.plot(window)
    else:
        flat = window.reshape(-1)
        if plot_style == 'dots':
            ax.plot(x, flat, 'o', markersize=markersize, linestyle='None')
        else:
            ax.plot(flat)

    overlay = panel_cfg.get('overlay')
    if overlay is not None:
        overlay_window = overlay['value']
        if overlay_window.ndim == 1:
            overlay_window = overlay_window.reshape(-1, 1)
        overlay_label = overlay.get('legend', ['overlay'])[0]
        overlay_x = np.arange(overlay_window.shape[0])
        overlay_style = overlay.get('plot_style', plot_style)
        overlay_markersize = overlay.get('markersize', markersize + 2)
        if overlay_style == 'dots':
            ax.plot(
                overlay_x, overlay_window[:, 0], 'o',
                label=overlay_label,
                color=overlay.get('color', 'black'),
                markersize=overlay_markersize,
                linestyle='None',
            )
        else:
            ax.plot(
                overlay_x, overlay_window[:, 0],
                label=overlay_label,
                color=overlay.get('color', 'black'),
                linewidth=overlay.get('linewidth', 3.0),
            )


def _set_panel_legend(ax, panel_cfg):
    if panel_cfg.get('overlay') is not None or panel_cfg.get('plot_columns'):
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    elif panel_cfg.get('legend') is not None:
        ax.legend(panel_cfg['legend'], loc="lower right", fontsize=8, ncol=2)


def draw_frames_with_info(values, images, experiment_title, start_idx=0, stop_idx=-1, WINDOWSIZE=34, pad=1):
    """
    Draw frames with overlaid value plots.
    
    values - dict{'value_name': {'value': array, 'legend': [...]}}
    images - list/array of frames (N, C, H, W) or (N, H, W, C)
    experiment_title - main title
    
    Returns list of images drawn with info.
    """
    # Create figure
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(experiment_title)
    fig.tight_layout(pad=pad)

    n_values = len(values.keys())
    gs = gridspec.GridSpec(n_values + 1, 5)

    axs = {}
    skip_v1_v2 = False
    
    for n_row in range(int(skip_v1_v2), n_values):
        value_name = list(values.keys())[n_row]
        axs[value_name] = plt.subplot(gs[n_row, :3])
        ymin, ymax = _panel_ylim(values[value_name], values[value_name]['value'])
        axs[value_name].set_ylim(bottom=ymin, top=ymax)

    image_axs = plt.subplot(gs[0:n_values, 3:])

    if stop_idx < 0:
        stop_idx = len(images)

    # Start loop
    results = []
    for idx in tqdm(range(start_idx, stop_idx), desc="Drawing frames"):
        # Prepare indices for window
        start = max(0, idx - WINDOWSIZE // 2)
        end = idx + WINDOWSIZE // 2
        dotpos = (end - start) // 2
        if idx - WINDOWSIZE // 2 < 0:
            dotpos = idx

        # Plot values on axes
        for value_name in axs:
            panel_cfg = values[value_name]
            val_data = panel_cfg['value']
            window = val_data[start:end]
            plot_cfg = panel_cfg
            if 'overlay' in panel_cfg:
                plot_cfg = dict(panel_cfg)
                overlay = dict(panel_cfg['overlay'])
                overlay['value'] = overlay['value'][start:end]
                plot_cfg['overlay'] = overlay
            ymin, ymax = _panel_ylim(panel_cfg, val_data)
            axs[value_name].set_ylim(bottom=ymin, top=ymax)
            axs[value_name].set_title(value_name, fontstyle='italic')
            _plot_panel(axs[value_name], window, plot_cfg)
            axs[value_name].axvline(x=dotpos, color='green')
            _set_panel_legend(axs[value_name], plot_cfg)

        # Plot image - handle different input formats
        img = images[idx]
        if len(img.shape) == 3:
            if img.shape[0] == 1:  # (1, H, W) - single channel
                img_display = img[0]
            elif img.shape[0] == 2:  # (2, H, W) - two channels, use first
                img_display = img[0]
            elif img.shape[0] == 3:  # (3, H, W) - RGB
                img_display = np.transpose(img, (1, 2, 0))
            elif img.shape[2] == 1:  # (H, W, 1) format
                img_display = img[:, :, 0]
            elif img.shape[2] in [2, 3, 4]:  # (H, W, C) format
                img_display = img  # Already in right format
            else:  # Unknown format, just use first channel
                img_display = img[0] if img.shape[0] < img.shape[2] else img[:, :, 0]
        elif len(img.shape) == 2:
            img_display = img
        else:
            img_display = img
            
        image_axs.imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)

        # Add drawing to results
        fig.tight_layout(pad=pad)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # Get RGB data from figure canvas (compatible with both old and new matplotlib)
        width, height = fig.canvas.get_width_height()
        if hasattr(fig.canvas, 'buffer_rgba'):
            # New matplotlib API - returns RGBA buffer
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape((height, width, 4))
            # Drop alpha channel to get RGB
            image_from_plot = buf[:, :, :3]
        elif hasattr(fig.canvas, 'tostring_rgb'):
            # Old matplotlib API
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape((height, width, 3))
        elif hasattr(fig.canvas, 'tostring_argb'):
            # Alternative old API - ARGB format
            buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf = buf.reshape((height, width, 4))
            # Drop alpha channel (first channel in ARGB)
            image_from_plot = buf[:, :, 1:]
        else:
            raise RuntimeError("Cannot get RGB data from figure canvas")
        results.append(image_from_plot.copy())
        
        # Clear for the next frame - use cla() and remove all artists
        for value_name in axs:
            axs[value_name].cla()
        image_axs.cla()

    plt.close(fig)
    return results


def download_eval_files(eval_folder, local_dir, server, username, remote_project_path, pkey_path="~/.ssh/id_rsa"):
    """
    Download evaluation files from remote server via SFTP.
    """
    if not HAS_PARAMIKO:
        print("[download] ERROR: paramiko is not installed. Cannot download from remote server.")
        print("[download] Install with: pip install paramiko")
        print("[download] Or use --no-download to process local files only.")
        return False
    
    print(f"[download] Connecting to {server} as {username}...")
    
    # Load private key
    pkey_path = os.path.expanduser(pkey_path)
    if not os.path.exists(pkey_path):
        print(f"[download] Warning: SSH key not found at {pkey_path}, trying default auth...")
        pkey = None
    else:
        try:
            pkey = paramiko.RSAKey.from_private_key(open(pkey_path))
        except Exception as e:
            print(f"[download] Could not load RSA key: {e}, trying default auth...")
            pkey = None
    
    # Connect via SSH
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    
    try:
        if pkey:
            ssh.connect(server, username=username, pkey=pkey,
                       look_for_keys=False, allow_agent=False,
                       disabled_algorithms={'pubkeys': ['rsa-sha2-512', 'rsa-sha2-256']})
        else:
            # Try with default auth
            ssh.connect(server, username=username)
    except Exception as e:
        print(f"[download] SSH connection failed: {e}")
        return False
    
    sftp = ssh.open_sftp()
    
    # List remote files
    remote_eval_path = f"{remote_project_path}/{eval_folder}"
    print(f"[download] Listing files in {remote_eval_path}...")
    
    try:
        all_filenames = sftp.listdir(remote_eval_path)
    except Exception as e:
        print(f"[download] Could not list remote directory: {e}")
        return False
    
    # Files we need to download (matching the old naming convention)
    needed_suffixes = [
        'Q11s.npy', 'Q22s.npy', 'Q21s.npy', 'aas.npy', 'oos.npy',
        'betas.npy', 'beta_logits.npy', 'beta_active.npy', 'beta_samples.npy',
        'Frames_normalized_orig.npy', 'Vs.npy', 'Vs2.npy',
        'rewards.npy', 'gs2.npy', 'gs1.npy', 'ss.npy', 'ss2.npy'
    ]
    
    # Find files to download
    to_download = []
    for filename in all_filenames:
        for suffix in needed_suffixes:
            if filename.endswith(suffix):
                to_download.append(filename)
                break
    
    print(f"[download] Found {len(to_download)} files to download out of {len(all_filenames)} total files")
    
    # Create local directory
    try:
        os.makedirs(local_dir, exist_ok=True)
        print(f"[download] Local directory: {local_dir}")
    except Exception as e:
        print(f"[download] Error creating local directory: {e}")
        return False
    
    # Download files
    for filename in to_download:
        local_path = os.path.join(local_dir, filename)
        remote_path = f"{remote_eval_path}/{filename}"
        
        # Skip if already exists
        if os.path.exists(local_path):
            print(f"[download] Skipping {filename} (already exists)")
            continue
        
        try:
            print(f"[download] Downloading {filename}...")
            sftp.get(remote_path, local_path)
        except Exception as e:
            print(f"[download] Error downloading {filename}: {e}")
    
    sftp.close()
    ssh.close()
    print("[download] Download complete!")
    return True


def load_eval_data(eval_folder, local_dir):
    """
    Load evaluation data from local .npy files.
    """
    data = {}
    
    # Check if local directory exists
    if not os.path.exists(local_dir):
        # If eval_folder is a path, try using it directly
        if os.path.exists(eval_folder):
            local_dir = eval_folder
            print(f"[load] Using eval_folder as local_dir: {local_dir}")
        else:
            print(f"[load] ERROR: Directory not found: {local_dir}")
            return data
    
    all_files = os.listdir(local_dir)
    
    # Mapping from file suffixes to data keys
    suffix_to_key = {
        'Q11s.npy': 'Q_int',      # Level 1 logits (intrinsic/mixed)
        'Q22s.npy': 'Q_ext',      # Level 2 logits (extrinsic)
        'aas.npy': 'action',      # Selected actions
        'oos.npy': 'option',      # NEW: option indices actually played
        'betas.npy': 'betas',     # Per-option termination betas
        'beta_logits.npy': 'beta_logits',  # Pre-sigmoid beta logits
        'beta_active.npy': 'beta_active',  # Active termination beta
        'beta_samples.npy': 'beta_samples',  # NEW: Bernoulli terminate samples
        'Vs.npy': 'Vs',           # Level 1 values
        'Vs2.npy': 'Vs2',         # Level 2 values
        'rewards.npy': 'rewards', # Rewards
        'Frames_normalized_orig.npy': 'frames',  # Normalized frames
        'ss.npy': 'ss',           # Level 1 features
        'ss2.npy': 'ss2',         # Level 2 features
    }
    
    for filename in all_files:
        for suffix, key in suffix_to_key.items():
            if filename.endswith(suffix):
                filepath = os.path.join(local_dir, filename)
                try:
                    loaded = np.load(filepath)
                    data[key] = loaded
                    print(f"[load] Loaded {key}: {loaded.shape} (from {filename})")
                except Exception as e:
                    print(f"[load] Could not load {filename}: {e}")
                break
    
    return data


def create_action_gif(data, eval_folder, local_dir, fps=3, start_idx=0, stop_idx=300, window_size=34):
    """
    Create actions visualization MP4 showing logits1, level2 logits, and sampled beta.
    """
    print("\n[draw] Creating actions visualization...")
    
    frames = data['frames']
    
    # Get action legend from environment (Pong has 6 actions)
    action_legend = ["noop", "fire", "right", "left", "rightfire", "leftfire"]
    
    # Prepare Q-values - handle different action spaces
    Q_int = data.get('Q_int', None)
    Q_ext = data.get('Q_ext', None)
    actions = data.get('action', None)
    options = data.get('option', None)
    betas = data.get('betas', None)
    beta_active = data.get('beta_active', None)
    beta_samples = data.get('beta_samples', None)
    
    if Q_int is not None:
        print(f"[draw] Q_int shape: {Q_int.shape}, range: [{np.min(Q_int):.2f}, {np.max(Q_int):.2f}]")
    if Q_ext is not None:
        print(f"[draw] Q_ext shape: {Q_ext.shape}, range: [{np.min(Q_ext):.2f}, {np.max(Q_ext):.2f}]")
    if options is not None:
        print(f"[draw] Options shape: {options.shape}, range: [{np.min(options)}, {np.max(options)}]")
    if beta_samples is not None:
        print(f"[draw] Beta_samples shape: {beta_samples.shape}, "
              f"mean terminate={np.mean(beta_samples):.3f}")
    if betas is not None:
        print(f"[draw] Betas shape: {betas.shape}, range: [{np.min(betas):.2f}, {np.max(betas):.2f}]")
    if beta_active is not None:
        print(f"[draw] Beta_active shape: {beta_active.shape}, range: [{np.min(beta_active):.2f}, {np.max(beta_active):.2f}]")
    if actions is not None:
        print(f"[draw] Actions shape: {actions.shape}")
    
    # Build values dict: logits1, level2 logits, beta sampled
    values_dict = {}
    if Q_int is not None:
        values_dict['logits1 (level1)'] = {
            'value': Q_int,
            'legend': action_legend[:Q_int.shape[1]] if len(Q_int.shape) > 1 else action_legend,
        }
    if Q_ext is not None:
        values_dict['logits2 (level2)'] = {
            'value': Q_ext,
            'legend': [f'opt{i}' for i in range(Q_ext.shape[1])],
            'plot_columns': True,
        }
    if options is not None:
        options_flat = np.squeeze(options)
        if options_flat.ndim == 1:
            options_flat = options_flat.reshape(-1, 1)
        n_opts = int(np.max(options_flat)) + 1 if options_flat.size else 1
        values_dict['option2 (played)'] = {
            'value': options_flat,
            'legend': ['option'],
            'ylim': (-0.5, max(n_opts - 0.5, 0.5)),
            'plot_style': 'dots',
            'markersize': 7,
        }
    if beta_samples is not None:
        beta_samples_flat = np.squeeze(beta_samples)
        if beta_samples_flat.ndim == 1:
            beta_samples_flat = beta_samples_flat.reshape(-1, 1)
        values_dict['beta (sampled terminate)'] = {
            'value': beta_samples_flat,
            'legend': ['terminate'],
            'ylim': (-0.1, 1.1),
            'plot_style': 'dots',
            'markersize': 7,
        }
    elif betas is not None:
        # Fallback for older evals without beta_samples.npy
        beta_panel = {
            'value': betas,
            'legend': [f'opt{i}' for i in range(betas.shape[1])],
            'ylim': (0.0, 1.0),
            'plot_columns': True,
            'plot_style': 'dots',
            'markersize': 5,
        }
        if beta_active is not None:
            beta_active_flat = np.squeeze(beta_active)
            if beta_active_flat.ndim == 1:
                beta_panel['overlay'] = {
                    'value': beta_active_flat,
                    'legend': ['beta_active (current option)'],
                    'color': 'black',
                    'plot_style': 'dots',
                    'markersize': 8,
                }
        values_dict['beta probabilities P(terminate | opt_i)'] = beta_panel
    elif beta_active is not None:
        beta_active_flat = np.squeeze(beta_active)
        if beta_active_flat.ndim == 1:
            beta_active_flat = beta_active_flat.reshape(-1, 1)
        values_dict['beta_active P(terminate)'] = {
            'value': beta_active_flat,
            'legend': ['beta_active'],
            'ylim': (0.0, 1.0),
            'plot_style': 'dots',
            'markersize': 6,
        }
    elif actions is not None:
        # Squeeze actions if needed
        if len(actions.shape) > 1:
            actions = np.squeeze(actions, axis=1)
        values_dict['action'] = {'value': actions.reshape(-1, 1), 'legend': action_legend}
    
    # Create frames
    dd = draw_frames_with_info(
        values_dict,
        frames,
        f'{eval_folder} - Actions',
        start_idx=start_idx,
        stop_idx=stop_idx,
        WINDOWSIZE=window_size
    )
    
    # Crop if needed (based on original code)
    CROP_BELOW = 1650
    dd = [k[:CROP_BELOW] for k in dd]
    
    # Save
    output_path = os.path.join(local_dir, "actions.mp4")
    print(f"[draw] Saving to {output_path}")
    imageio.mimsave(output_path, dd, fps=fps)
    print(f"[draw] Actions GIF saved!")
    
    del dd


def create_values_gif(data, eval_folder, local_dir, fps=3, start_idx=0, stop_idx=300, window_size=34):
    """
    Create values visualization GIF showing V1, V2 and rewards.
    """
    print("\n[draw] Creating values visualization...")
    
    frames = data['frames']
    Vs = data.get('Vs', None)
    Vs2 = data.get('Vs2', None)
    rewards = data.get('rewards', None)
    
    if Vs is not None:
        print(f"[draw] Vs shape: {Vs.shape}, range: [{np.min(Vs):.2f}, {np.max(Vs):.2f}]")
        # Expand dims if needed
        if len(Vs.shape) == 1:
            Vs = np.expand_dims(Vs, axis=1)
    if Vs2 is not None:
        print(f"[draw] Vs2 shape: {Vs2.shape}, range: [{np.min(Vs2):.2f}, {np.max(Vs2):.2f}]")
        if len(Vs2.shape) == 1:
            Vs2 = np.expand_dims(Vs2, axis=1)
    if rewards is not None:
        print(f"[draw] Rewards shape: {rewards.shape}")
        print(f"[draw] Reward stats: mean={np.mean(rewards):.2f}, max={np.max(rewards):.2f}, min={np.min(rewards):.2f}")
    
    # Build values dict
    values_dict = {}
    
    # Combine V1 and V2 with clear labels
    if Vs is not None and Vs2 is not None:
        values_dict['V1_and_V2'] = {
            'value': np.hstack([Vs, Vs2]),
            'legend': ["V1 (level1)", "V2 (level2)"]
        }
    elif Vs is not None:
        values_dict['V1 (level1)'] = {'value': Vs}
    elif Vs2 is not None:
        values_dict['V2 (level2)'] = {'value': Vs2}
    
    if rewards is not None:
        values_dict['reward'] = {'value': rewards}
    
    # Create frames
    dd = draw_frames_with_info(
        values_dict,
        frames,
        f'{eval_folder} - Values',
        start_idx=start_idx,
        stop_idx=stop_idx,
        WINDOWSIZE=window_size
    )
    
    # Crop if needed (based on original code)
    CROP_BELOW = 1450
    dd = [k[:CROP_BELOW] for k in dd]
    
    # Save
    output_path = os.path.join(local_dir, "VS_short.mp4")
    print(f"[draw] Saving to {output_path}")
    imageio.mimsave(output_path, dd, fps=fps)
    print(f"[draw] Values GIF saved!")
    
    del dd


def main():
    parser = argparse.ArgumentParser(
        description='Download eval results and create visualization GIFs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python src/draw_eval_gifs.py Eval_2024-12-04_21:58:10_cosineFix2_try2.468102
        """
    )
    parser.add_argument('eval_folder', help='Name of the eval folder on remote server')
    parser.add_argument('--local-dir', default=None,
                       help='Local directory to download files (default: same as eval folder name)')
    parser.add_argument('--server', default='ui4.computing.kiae.ru',
                       help='Server SSH name/address')
    parser.add_argument('--username', default='aamore',
                       help='SSH username')
    parser.add_argument('--remote-path', default='/home/users/aamore/rl_a3c_slurm/',
                       help='Remote project path where eval folders are located')
    parser.add_argument('--ssh-key', default='~/.ssh/id_rsa',
                       help='Path to SSH private key')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip download (use local files if they exist)')
    parser.add_argument('--fps', type=int, default=3,
                       help='FPS for output videos (default: 3)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Start frame index (default: 0)')
    parser.add_argument('--stop-idx', type=int, default=300,
                       help='Stop frame index (default: 300, -1 for all)')
    parser.add_argument('--window-size', type=int, default=34,
                       help='Window size for value plots (default: 34)')
    
    args = parser.parse_args()
    
    # Determine local directory
    local_dir = args.local_dir if args.local_dir else args.eval_folder
    
    print(f"="*60)
    print(f"Evaluation Folder: {args.eval_folder}")
    print(f"Local Directory: {local_dir}")
    print(f"="*60)
    
    # Download files if requested
    if not args.no_download:
        success = download_eval_files(
            args.eval_folder,
            local_dir,
            args.server,
            args.username,
            args.remote_path,
            args.ssh_key
        )
        if not success:
            print("[main] Download failed, but will try to use local files if available...")
    else:
        print("[main] Skipping download (--no-download specified)")
    
    # Load data
    data = load_eval_data(args.eval_folder, local_dir)
    
    if not data:
        print("[main] No data loaded! Exiting.")
        return
    
    if 'frames' not in data:
        print("[main] No frames data found! Cannot create GIFs.")
        return
    
    # Adjust stop_idx if -1 (meaning all frames)
    if args.stop_idx < 0:
        args.stop_idx = len(data['frames'])
    
    print(f"\n[main] Creating GIFs for frames {args.start_idx} to {args.stop_idx}")
    
    # Create GIFs
    try:
        create_action_gif(data, args.eval_folder, local_dir, 
                         fps=args.fps, start_idx=args.start_idx, 
                         stop_idx=args.stop_idx, window_size=args.window_size)
    except Exception as e:
        print(f"[main] Error creating action GIF: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        create_values_gif(data, args.eval_folder, local_dir,
                         fps=args.fps, start_idx=args.start_idx,
                         stop_idx=args.stop_idx, window_size=args.window_size)
    except Exception as e:
        print(f"[main] Error creating values GIF: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[main] Done! Output files in: {local_dir}")
    print(f"  - {os.path.join(local_dir, 'actions.mp4')}")
    print(f"  - {os.path.join(local_dir, 'VS_short.mp4')}")


if __name__ == '__main__':
    main()
