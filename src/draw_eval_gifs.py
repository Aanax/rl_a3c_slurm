"""
draw_eval_gifs.py — Download evaluation results from remote server and create visualization GIFs.

This script downloads evaluation artifacts from a remote server (paramiko SFTP by
default; OpenSSH rsync with --rsync for hosts that need ssh-rsa),
then creates two MP4 videos:
  1. actions.mp4 - Configurable panels (max 4) overlaid on frames
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
    --rsync                   Download via OpenSSH rsync instead of paramiko
    --no-download             Skip download (use local files if they exist)
    --fps FPS                 FPS for output GIFs (default: 3)
    --start-idx START         Start frame index (default: 0)
    --stop-idx STOP           Stop frame index (default: 300, -1 for all)
    --window-size SIZE        Window size for value plots (default: 34)
    --actions {auto,pong,pacman}
                              Action names for logits1 (default: auto from Q11s width:
                              6=Pong, 9=MsPacman; leftover columns labeled aN)
    --panels P1 [P2 P3 P4]    Up to 4 params to draw on actions.mp4
                              Default: beta2 option2 action
                              MP4s saved under panels_<p1>_<p2>_.../ (no overwrite)

Available --panels names:
    beta2, option2, action, logits1, logits2,
    beta_samples, betas, beta_active, terminated1, terminated2

Example:
    python src/draw_eval_gifs.py Eval_2026-03-29_00:21:09_PongNoFrameskip-v4 --no-download
    python src/draw_eval_gifs.py Eval_xxx --panels beta2 option2 action --no-download
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
import shlex

# Optional paramiko import (used for default SFTP download)
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    paramiko = None

PARAMIKO_DEFAULTS = {
    'server': 'ui4.computing.kiae.ru',
    'remote_path': '/home/users/aamore/rl_a3c_slurm/',
    'ssh_key': '~/.ssh/id_rsa',
}
RSYNC_DEFAULTS = {
    'server': 'ki',
    'remote_path': '/s/ls4/users/aamore/rl_a3c_pytorch/',
    'ssh_key': '~/.ssh/id_rsa2',
}


DEFAULT_ACTION_PANELS = ['beta2', 'option2', 'action']
MAX_ACTION_PANELS = 4

PONG_ACTIONS = ["noop", "fire", "right", "left", "rightfire", "leftfire"]
PACMAN_ACTIONS = [
    "noop", "up", "right", "left", "down",
    "upright", "upleft", "downright", "downleft",
]
ACTION_LEGENDS = {
    'pong': PONG_ACTIONS,
    'pacman': PACMAN_ACTIONS,
}

# Aliases -> canonical panel keys
PANEL_ALIASES = {
    'beta2': 'beta2',
    'option': 'option2',
    'option2': 'option2',
    'option2_played': 'option2',
    'action': 'action',
    'action_chosen': 'action',
    'logits1': 'logits1',
    'logits2': 'logits2',
    'q11': 'logits1',
    'q22': 'logits2',
    'beta_samples': 'beta_samples',
    'betas': 'betas',
    'beta_active': 'beta_active',
    'terminated1': 'terminated1',
    'terminated2': 'terminated2',
}


def _n_discrete_actions(data):
    """Number of level-1 actions from Q11s width, else max(action)+1."""
    Q_int = data.get('Q_int')
    if Q_int is not None and Q_int.ndim > 1 and Q_int.shape[1] > 0:
        return int(Q_int.shape[1])
    actions = data.get('action')
    if actions is not None and np.size(actions):
        return int(np.max(actions)) + 1
    return 0


def resolve_action_legend(n_actions, actions_arg='auto'):
    """Return n_actions labels. auto: 6=Pong, 9=MsPacman; extras are aN."""
    if n_actions <= 0:
        return []
    if actions_arg in ACTION_LEGENDS:
        names = list(ACTION_LEGENDS[actions_arg])
        source = actions_arg
    elif n_actions == len(PONG_ACTIONS):
        names = list(PONG_ACTIONS)
        source = 'pong (auto)'
    elif n_actions == len(PACMAN_ACTIONS):
        names = list(PACMAN_ACTIONS)
        source = 'pacman (auto)'
    else:
        names = []
        source = f'generic ({n_actions} actions)'
    legend = [names[i] if i < len(names) else f'a{i}' for i in range(n_actions)]
    print(f"[draw] Action legend ({source}): {legend}")
    return legend


def _as_col(arr):
    """Squeeze and reshape to (N, 1) for single-series panels."""
    flat = np.squeeze(arr)
    if flat.ndim == 0:
        flat = flat.reshape(1)
    if flat.ndim == 1:
        return flat.reshape(-1, 1)
    return flat


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

    n_images = len(images)
    if stop_idx < 0 or stop_idx > n_images:
        stop_idx = n_images
    start_idx = max(0, min(start_idx, n_images))

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


def _openssh_rsync_rsh(username, pkey_path):
    """ssh command for rsync -e. Enables ssh-rsa (old KIAE host keys)."""
    ssh_cmd = [
        "ssh",
        "-o", "HostKeyAlgorithms=+ssh-rsa",
        "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    ]
    pkey_path = os.path.expanduser(pkey_path) if pkey_path else ""
    if pkey_path and os.path.isfile(pkey_path):
        ssh_cmd.extend(["-i", pkey_path])
    if username:
        ssh_cmd.extend(["-l", username])
    return " ".join(shlex.quote(part) for part in ssh_cmd)


def download_eval_files_rsync(eval_folder, local_dir, server, username, remote_project_path, pkey_path):
    """Download evaluation .npy files via OpenSSH rsync (opt-in with --rsync)."""
    remote_eval_path = f"{remote_project_path.rstrip('/')}/{eval_folder.strip('/')}"
    remote_spec = f"{server}:{remote_eval_path}/"
    local_spec = local_dir.rstrip("/") + "/"

    try:
        os.makedirs(local_dir, exist_ok=True)
    except Exception as e:
        print(f"[download] Error creating local directory: {e}")
        return False

    rsh = _openssh_rsync_rsh(username, pkey_path)
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", rsh,
        "--include", "*/",
        "--include", "*.npy",
        "--exclude", "*",
        remote_spec,
        local_spec,
    ]
    print(f"[download] Connecting via OpenSSH rsync to {server} as {username}...")
    print(f"[download] Remote: {remote_spec}")
    print(f"[download] Local:  {local_spec}")
    print(f"[download] Running: {' '.join(shlex.quote(c) for c in cmd)}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[download] rsync failed (exit {result.returncode}).")
        print("[download] If the remote path is wrong, pass --remote-path.")
        return False

    print("[download] Download complete!")
    return True


def download_eval_files_paramiko(eval_folder, local_dir, server, username, remote_project_path, pkey_path="~/.ssh/id_rsa"):
    """
    Download evaluation files from remote server via SFTP.
    """
    if not HAS_PARAMIKO:
        print("[download] ERROR: paramiko is not installed. Cannot download from remote server.")
        print("[download] Install with: pip install paramiko")
        print("[download] Or use --rsync (OpenSSH) / --no-download.")
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
        'beta2s.npy', 'terminated1s.npy', 'terminated2s.npy',
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


def download_eval_files(eval_folder, local_dir, server, username, remote_project_path, pkey_path, use_rsync=False):
    """Download eval files: paramiko SFTP by default, OpenSSH rsync if use_rsync."""
    if use_rsync:
        return download_eval_files_rsync(
            eval_folder, local_dir, server, username, remote_project_path, pkey_path
        )
    return download_eval_files_paramiko(
        eval_folder, local_dir, server, username, remote_project_path, pkey_path
    )


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
    # Longer / more-specific suffixes first so e.g. beta_active matches before beta.
    suffix_to_key = [
        ('Frames_normalized_orig.npy', 'frames'),
        ('beta_logits.npy', 'beta_logits'),
        ('beta_active.npy', 'beta_active'),
        ('beta_samples.npy', 'beta_samples'),
        ('terminated1s.npy', 'terminated1'),
        ('terminated2s.npy', 'terminated2'),
        ('beta2s.npy', 'beta2'),
        ('betas.npy', 'betas'),
        ('Q11s.npy', 'Q_int'),
        ('Q22s.npy', 'Q_ext'),
        ('aas.npy', 'action'),
        ('oos.npy', 'option'),
        ('Vs2.npy', 'Vs2'),
        ('Vs.npy', 'Vs'),
        ('rewards.npy', 'rewards'),
        ('ss2.npy', 'ss2'),
        ('ss.npy', 'ss'),
    ]
    
    for filename in all_files:
        for suffix, key in suffix_to_key:
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


def normalize_panel_names(panel_names):
    """Validate/normalize panel names; at most MAX_ACTION_PANELS."""
    if not panel_names:
        panel_names = list(DEFAULT_ACTION_PANELS)
    if len(panel_names) > MAX_ACTION_PANELS:
        raise ValueError(
            f"At most {MAX_ACTION_PANELS} panels allowed, got {len(panel_names)}: {panel_names}"
        )
    normalized = []
    unknown = []
    for name in panel_names:
        key = PANEL_ALIASES.get(name.lower().strip())
        if key is None:
            unknown.append(name)
        else:
            normalized.append(key)
    if unknown:
        available = ', '.join(sorted(set(PANEL_ALIASES.keys())))
        raise ValueError(
            f"Unknown panel name(s): {unknown}. Available: {available}"
        )
    return normalized


def build_available_panels(data, actions_arg='auto'):
    """Build a dict of panel_key -> panel_cfg from loaded eval arrays."""
    action_legend = resolve_action_legend(_n_discrete_actions(data), actions_arg)
    panels = {}

    Q_int = data.get('Q_int')
    Q_ext = data.get('Q_ext')
    actions = data.get('action')
    options = data.get('option')
    betas = data.get('betas')
    beta_active = data.get('beta_active')
    beta_samples = data.get('beta_samples')
    beta2 = data.get('beta2')
    terminated1 = data.get('terminated1')
    terminated2 = data.get('terminated2')

    if Q_int is not None:
        n_cols = Q_int.shape[1] if Q_int.ndim > 1 else 1
        panels['logits1'] = {
            'value': Q_int,
            'legend': action_legend[:n_cols] if action_legend else [f'a{i}' for i in range(n_cols)],
            'plot_columns': n_cols > 1,
            'title': 'logits1 (level1)',
        }
    if Q_ext is not None:
        panels['logits2'] = {
            'value': Q_ext,
            'legend': [f'opt{i}' for i in range(Q_ext.shape[1])],
            'plot_columns': True,
            'title': 'logits2 (level2)',
        }
    if options is not None:
        options_flat = _as_col(options)
        n_opts = int(np.max(options_flat)) + 1 if options_flat.size else 1
        panels['option2'] = {
            'value': options_flat,
            'legend': ['option'],
            'ylim': (-0.5, max(n_opts - 0.5, 0.5)),
            'plot_style': 'dots',
            'markersize': 7,
            'title': 'option2 (played)',
        }
    if actions is not None:
        actions_flat = _as_col(actions)
        n_acts = int(np.max(actions_flat)) + 1 if actions_flat.size else 1
        panels['action'] = {
            'value': actions_flat,
            'legend': ['action'],
            'ylim': (-0.5, max(n_acts - 0.5, 0.5)),
            'plot_style': 'dots',
            'markersize': 7,
            'title': 'action (chosen)',
        }
    if beta2 is not None:
        panels['beta2'] = {
            'value': _as_col(beta2),
            'legend': ['beta2'],
            'ylim': (0.0, 1.0),
            'plot_style': 'dots',
            'markersize': 6,
            'title': 'beta2',
        }
    elif beta_active is not None:
        # Older evals: only beta_active (== beta2 of current option)
        panels['beta2'] = {
            'value': _as_col(beta_active),
            'legend': ['beta2'],
            'ylim': (0.0, 1.0),
            'plot_style': 'dots',
            'markersize': 6,
            'title': 'beta2 (beta_active)',
        }
    if terminated1 is not None:
        panels['terminated1'] = {
            'value': _as_col(terminated1),
            'legend': ['terminated1'],
            'ylim': (-0.1, 1.1),
            'plot_style': 'dots',
            'markersize': 7,
            'title': 'terminated1',
        }
    if terminated2 is not None:
        panels['terminated2'] = {
            'value': _as_col(terminated2),
            'legend': ['terminated2'],
            'ylim': (-0.1, 1.1),
            'plot_style': 'dots',
            'markersize': 7,
            'title': 'terminated2',
        }
    if beta_samples is not None:
        panels['beta_samples'] = {
            'value': _as_col(beta_samples),
            'legend': ['terminate'],
            'ylim': (-0.1, 1.1),
            'plot_style': 'dots',
            'markersize': 7,
            'title': 'beta (sampled terminate)',
        }
        # Alias for older naming when terminated2 file is absent
        if 'terminated2' not in panels:
            panels['terminated2'] = dict(panels['beta_samples'])
            panels['terminated2']['title'] = 'terminated2 (beta_samples)'
    if betas is not None:
        beta_panel = {
            'value': betas,
            'legend': [f'opt{i}' for i in range(betas.shape[1])] if betas.ndim > 1 else ['beta'],
            'ylim': (0.0, 1.0),
            'plot_columns': betas.ndim > 1 and betas.shape[1] > 1,
            'plot_style': 'dots',
            'markersize': 5,
            'title': 'beta probabilities P(terminate | opt_i)',
        }
        if beta_active is not None:
            beta_panel['overlay'] = {
                'value': _as_col(beta_active).reshape(-1),
                'legend': ['beta_active (current option)'],
                'color': 'black',
                'plot_style': 'dots',
                'markersize': 8,
            }
        panels['betas'] = beta_panel
    if beta_active is not None:
        panels['beta_active'] = {
            'value': _as_col(beta_active),
            'legend': ['beta_active'],
            'ylim': (0.0, 1.0),
            'plot_style': 'dots',
            'markersize': 6,
            'title': 'beta_active P(terminate)',
        }

    return panels


def select_panels(available_panels, panel_names):
    """Pick requested panels in order; skip missing with a warning."""
    values_dict = {}
    for name in panel_names:
        if name not in available_panels:
            print(f"[draw] Warning: panel '{name}' not available in loaded data, skipping")
            continue
        cfg = dict(available_panels[name])
        title = cfg.pop('title', name)
        values_dict[title] = cfg
    return values_dict


def panels_output_dirname(panel_names):
    """Folder name for a panel combo, e.g. panels_beta2_option2_action."""
    return 'panels_' + '_'.join(panel_names)


def make_panels_output_dir(local_dir, panel_names):
    """Create and return local_dir/panels_<p1>_<p2>_... for this --panels run."""
    out_dir = os.path.join(local_dir, panels_output_dirname(panel_names))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def create_action_gif(
    data,
    eval_folder,
    output_dir,
    fps=3,
    start_idx=0,
    stop_idx=300,
    window_size=34,
    panels=None,
    actions='auto',
):
    """
    Create actions visualization MP4 with up to 4 selected panels.
    Saves to output_dir/actions.mp4 (caller should pass a panels-specific folder).
    """
    print("\n[draw] Creating actions visualization...")
    
    frames = data['frames']
    panel_names = normalize_panel_names(panels)
    print(f"[draw] Requested panels: {panel_names}")

    available = build_available_panels(data, actions_arg=actions)
    print(f"[draw] Available panels: {sorted(available.keys())}")

    for key, arr in [
        ('Q_int', data.get('Q_int')),
        ('Q_ext', data.get('Q_ext')),
        ('option', data.get('option')),
        ('action', data.get('action')),
        ('beta2', data.get('beta2')),
        ('beta_samples', data.get('beta_samples')),
        ('betas', data.get('betas')),
        ('beta_active', data.get('beta_active')),
        ('terminated1', data.get('terminated1')),
        ('terminated2', data.get('terminated2')),
    ]:
        if arr is not None:
            print(f"[draw] {key} shape: {arr.shape}")

    values_dict = select_panels(available, panel_names)
    if not values_dict:
        raise RuntimeError(
            f"None of the requested panels are available: {panel_names}. "
            f"Available: {sorted(available.keys())}"
        )
    print(f"[draw] Drawing panels: {list(values_dict.keys())}")
    
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
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "actions.mp4")
    print(f"[draw] Saving to {output_path}")
    imageio.mimsave(output_path, dd, fps=fps)
    print(f"[draw] Actions GIF saved!")
    
    del dd


def create_values_gif(data, eval_folder, output_dir, fps=3, start_idx=0, stop_idx=300, window_size=34):
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
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "VS_short.mp4")
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
    python src/draw_eval_gifs.py Eval_xxx --panels beta2 option2 action --no-download
    python src/draw_eval_gifs.py Eval_xxx --rsync --panels beta2 option2 action

MP4s are saved under Eval_xxx/panels_<p1>_<p2>_.../ so different --panels
combos do not overwrite each other.

Available --panels: beta2, option2, action, logits1, logits2,
                    beta_samples, betas, beta_active, terminated1, terminated2
        """
    )
    parser.add_argument('eval_folder', help='Name of the eval folder on remote server')
    parser.add_argument('--local-dir', default=None,
                       help='Local directory to download files (default: same as eval folder name)')
    parser.add_argument('--server', default=None,
                       help='SSH host (paramiko: ui4.computing.kiae.ru; --rsync: ki)')
    parser.add_argument('--username', default='aamore',
                       help='SSH username')
    parser.add_argument('--remote-path', default=None,
                       help='Remote project path where eval folders are located')
    parser.add_argument('--ssh-key', default=None,
                       help='Path to SSH private key')
    parser.add_argument('--rsync', action='store_true',
                       help='Download via OpenSSH rsync instead of paramiko (for ssh-rsa hosts)')
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
    parser.add_argument(
        '--actions',
        choices=['auto', 'pong', 'pacman'],
        default='auto',
        help=(
            'Action names for logits1 (default: auto). '
            'auto uses 6=Pong, 9=MsPacman from Q11s width; '
            'extra columns are labeled aN'
        ),
    )
    parser.add_argument(
        '--panels',
        nargs='+',
        default=None,
        metavar='PARAM',
        help=(
            f'Up to {MAX_ACTION_PANELS} params to draw on actions.mp4 '
            f'(default: {" ".join(DEFAULT_ACTION_PANELS)})'
        ),
    )
    
    args = parser.parse_args()
    
    # Determine local directory
    local_dir = args.local_dir if args.local_dir else args.eval_folder
    
    try:
        panel_names = normalize_panel_names(args.panels)
    except ValueError as e:
        print(f"[main] ERROR: {e}")
        return
    
    # npy inputs stay in local_dir; mp4s go to a panels-specific subfolder
    out_dir = make_panels_output_dir(local_dir, panel_names)

    print(f"="*60)
    print(f"Evaluation Folder: {args.eval_folder}")
    print(f"Local Directory: {local_dir}")
    print(f"Action panels: {panel_names}")
    print(f"Output Directory: {out_dir}")
    print(f"="*60)
    
    # Download files if requested
    if not args.no_download:
        ssh_defaults = RSYNC_DEFAULTS if args.rsync else PARAMIKO_DEFAULTS
        server = args.server or ssh_defaults['server']
        remote_path = args.remote_path or ssh_defaults['remote_path']
        ssh_key = args.ssh_key or ssh_defaults['ssh_key']
        success = download_eval_files(
            args.eval_folder,
            local_dir,
            server,
            args.username,
            remote_path,
            ssh_key,
            use_rsync=args.rsync,
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
    
    # Clamp to available frames (-1 or a default larger than the episode)
    n_frames = len(data['frames'])
    if args.stop_idx < 0 or args.stop_idx > n_frames:
        args.stop_idx = n_frames
    args.start_idx = max(0, min(args.start_idx, n_frames))
    
    print(f"\n[main] Creating GIFs for frames {args.start_idx} to {args.stop_idx}")
    
    # Create GIFs
    try:
        create_action_gif(
            data, args.eval_folder, out_dir,
            fps=args.fps, start_idx=args.start_idx,
            stop_idx=args.stop_idx, window_size=args.window_size,
            panels=panel_names,
            actions=args.actions,
        )
    except Exception as e:
        print(f"[main] Error creating action GIF: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        create_values_gif(data, args.eval_folder, out_dir,
                         fps=args.fps, start_idx=args.start_idx,
                         stop_idx=args.stop_idx, window_size=args.window_size)
    except Exception as e:
        print(f"[main] Error creating values GIF: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[main] Done! Output files in: {out_dir}")
    print(f"  - {os.path.join(out_dir, 'actions.mp4')}")
    print(f"  - {os.path.join(out_dir, 'VS_short.mp4')}")


if __name__ == '__main__':
    main()
