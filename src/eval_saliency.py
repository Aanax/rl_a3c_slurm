"""
eval_saliency.py — SARFA attention-map evaluation for Hierarchial_memory_memrelu.

Loads a trained .pth checkpoint, rolls out a greedy policy on an Atari
environment, computes a SARFA saliency map for every collected frame via
Gaussian-blur occlusion, and saves per-frame PNG overlays plus a compiled GIF.

Usage:
    python src/eval_saliency.py \\
        --config  configs/run_config_our5.ini \\
        --model-path ./trained_models/best_model.pth \\
        --num-frames 200 \\
        --density 5 \\
        --blur-radius 5 \\
        --output-dir ./saliency_output/pong_memrelu


sbatch slurm_scripts/run_saliency.sh \
    configs/run_config_our5.ini \
    trained_models/PongNoFrameskip-v4_score21_step12345.dat

No files under src/ or configs/ are modified by this script.
"""

from __future__ import print_function, division

import os
import sys
import json
import argparse
import configparser
import contextlib
import io

import numpy as np
import torch
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from cv2 import resize, INTER_LINEAR
import imageio

# ── path setup ────────────────────────────────────────────────────────────────
# This file lives in src/; insert src/ so `model` and `environment` resolve,
# and insert sarfa-saliency/ so `sarfa_saliency` resolves.
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_SARFA_DIR = os.path.join(_ROOT_DIR, 'sarfa-saliency')
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _SARFA_DIR)

import model as model_module                              # src/model.py
from environment import atari_env                        # src/environment.py
from sarfa_saliency import computeSaliencyUsingSarfa     # sarfa-saliency/sarfa_saliency.py


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Config Parsing & Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """
    Parse CLI flags and merge with the .ini run-config (identical fields to
    main.py).  The config file supplies all model/env hyperparameters; the CLI
    adds eval-only knobs that have no training equivalent.
    """
    p = argparse.ArgumentParser(
        description='SARFA saliency evaluation for Hierarchial_memory_memrelu'
    )
    # eval-only CLI flags
    p.add_argument('--config',      required=True,
                   help='Path to .ini run config (e.g. configs/run_config_our5.ini)')
    p.add_argument('--model-path',  required=True,
                   help='Path to .pth weights checkpoint')
    p.add_argument('--num-frames',  type=int,   default=200,
                   help='Number of frames to evaluate (default: 200)')
    p.add_argument('--density',     type=int,   default=5,
                   help='Perturbation grid stride in pixels (default: 5)')
    p.add_argument('--blur-radius', type=float, default=5.0,
                   help='Gaussian sigma for occlusion mask (default: 5.0)')
    p.add_argument('--output-dir',  default='./saliency_output',
                   help='Root directory for output PNGs and GIF (default: ./saliency_output)')
    p.add_argument('--alpha',       type=float, default=0.5,
                   help='Heatmap overlay opacity 0-1 (default: 0.5)')
    p.add_argument('--level',       choices=['a1', 'a2'], default='a1',
                   help='Level to monitor for saliency (a1 or a2, default: a1)')
    cli = p.parse_args()

    # ── read .ini — mirrors every field that main.py reads ───────────────────
    cfg = configparser.ConfigParser()
    cfg.read(cli.config)

    args = argparse.Namespace()

    args.env                       = cfg.get('DEFAULT', 'env')
    args.hidden_size               = cfg.getint('DEFAULT',   'hidden_size',               fallback=1024)
    args.gamma                     = cfg.getfloat('DEFAULT', 'gamma',                     fallback=0.9)
    args.gamma_memory              = cfg.getfloat('DEFAULT', 'gamma_memory',              fallback=0.9)
    args.gamma2                    = cfg.getfloat('DEFAULT', 'gamma2',                    fallback=0.99)
    args.tau                       = cfg.getfloat('DEFAULT', 'tau',                       fallback=1.0)
    args.skip_rate                 = cfg.getint('DEFAULT',   'skip_rate',                 fallback=4)
    args.max_episode_length        = cfg.getint('DEFAULT',   'max_episode_length',        fallback=10000)
    args.input_normalization_class = cfg.get('DEFAULT',      'input_normalization_class')
    args.model_type                = cfg.get('DEFAULT',      'model_type')
    args.env_config                = cfg.get('DEFAULT',      'env_config',
                                             fallback='configs/envs_config.json')
    args.normalization_alpha       = cfg.getfloat('DEFAULT', 'normalization_alpha',       fallback=0.9999)
    # these flags are training-only; fix them for eval
    args.monitor_s   = False
    args.use_rmsnorm = False
    gpu_str = cfg.get('DEFAULT', 'gpu_ids', fallback='-1')
    args.gpu_ids = [int(x.strip()) for x in gpu_str.split(',') if x.strip()]

    # ── attach eval-only extras ───────────────────────────────────────────────
    args.model_path  = cli.model_path
    args.num_frames  = cli.num_frames
    args.density     = cli.density
    args.blur_radius = cli.blur_radius
    args.output_dir  = cli.output_dir
    args.alpha       = cli.alpha
    args.level       = cli.level

    return args


def load_model_and_env(args):
    """
    Instantiate the Atari environment (via the existing atari_env wrapper) and
    load model weights from the .pth checkpoint.

    env_key is derived from the env-id so that the correct crop config is
    looked up in envs_config.json:
        'PongNoFrameskip-v4' → 'Pong'
    """
    env_conf = json.load(open(args.env_config))
    env_key  = args.env.split('-')[0].replace('NoFrameskip', '').replace('Deterministic', '')
    conf     = env_conf.get(env_key, env_conf['Default'])

    # Suppress atari_env's startup prints (gym version, "ENV was realdone", etc.)
    with contextlib.redirect_stdout(io.StringIO()):
        env = atari_env(args.env, conf, args)

    num_inputs = env.observation_space.shape[0]  # 2 for NormalizedEnvGameNormNoNormDiffOnNorm

    model_cls  = getattr(model_module, args.model_type)
    net        = model_cls(num_inputs, env.action_space, args)
    state_dict = torch.load(args.model_path, map_location='cpu')
    net.load_state_dict(state_dict)
    net.eval()

    print(f"[eval_saliency] Model       : {args.model_type}")
    print(f"[eval_saliency] Env         : {args.env}  (num_inputs={num_inputs})")
    print(f"[eval_saliency] Actions     : {env.action_space.n}")
    print(f"[eval_saliency] gamma       : {args.gamma}   gamma_memory: {args.gamma_memory}")
    print(f"[eval_saliency] Weights     : {args.model_path}")
    print(f"[eval_saliency] Density     : {args.density}  blur_radius: {args.blur_radius}")

    return net, env


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Policy Rollout with Memory Management
# ══════════════════════════════════════════════════════════════════════════════

def _reset_model_memory(net):
    """Zero out Hierarchial_memory_memrelu's running internal state."""
    net.running_mem = torch.zeros((1, 64, 4, 4))
    net.prev_x_conv = torch.zeros((1, 64, 4, 4))


def _get_logits(net, obs_t):
    """
    One forward pass; returns both level-1 and level-2 actor logits from
    the 8-tuple output.
    Tuple layout: (V1, a1, hx, cx, None, None, V2, a2_logits)
    stdout is suppressed to hide the debug print inside
    EncoderRules234_2_mem.forward() without touching src/model.py.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        with torch.no_grad():
            out = net(obs_t, torch.zeros(1), torch.zeros(1))
    a1_logits = out[1].squeeze(0)   # shape: (env.action_space.n,)
    a2_logits = out[7].squeeze(0)   # shape: (a2 action size)
    return a1_logits, a2_logits


def collect_rollout(net, env, num_frames, level):
    """
    Greedy policy rollout for num_frames steps.

    Returns five parallel lists:
        frames_rgb   — (H_raw, W_raw, 3) uint8 raw Atari frames
        obs_list     — (2, 80, 80) float32 processed observations
        logits_list  — (num_actions,) float32 actor logits
        action_list  — int chosen action indices
        memory_list  — tuples of (running_mem, prev_x_conv) after each step

    Model memory is reset at episode boundaries. After each forward pass,
    we snapshot memory so that saliency scans for each frame use the correct
    temporal state (not the end-of-rollout state).
    """
    frames_rgb, obs_list, logits_list, action_list, memory_list = [], [], [], [], []

    obs = env.reset()
    _reset_model_memory(net)

    print(f"[eval_saliency] Collecting {num_frames} rollout frames…")
    for step in range(num_frames):
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)   # (1, 2, 80, 80)
        a1_logits, a2_logits = _get_logits(net, obs_t)   # advances running_mem
        action = a1_logits.argmax().item()                # env action always based on a1
        logits = a1_logits if level == 'a1' else a2_logits

        # Capture raw RGB and processed obs
        frames_rgb.append(env.unwrapped.ale.getScreenRGB().copy())
        obs_list.append(obs.copy())
        logits_list.append(logits.numpy().copy())
        action_list.append(action)

        # SNAPSHOT: save memory AFTER this forward pass (for saliency of this frame)
        memory_list.append(_snapshot_memory(net))

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
            _reset_model_memory(net)

    print(f"[eval_saliency] Rollout complete ({num_frames} frames).")
    return frames_rgb, obs_list, logits_list, action_list, memory_list


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — SARFA Perturbation Loop (with Memory Snapshot / Restore)
# ══════════════════════════════════════════════════════════════════════════════

def _get_mask(center, size, r):
    """
    Soft circular occlusion mask, normalized to [0, 1].
        center : [row, col] — patch centre in pixel coords
        size   : [H, W]
        r      : Gaussian sigma (controls patch softness)
    """
    y, x = np.ogrid[-center[0]:size[0]-center[0],
                    -center[1]:size[1]-center[1]]
    keep = (x*x + y*y) <= 1
    mask = np.zeros(size, dtype=np.float32)
    mask[keep] = 1.0
    mask = gaussian_filter(mask, sigma=r)
    return mask / mask.max()


def _occlude(obs, mask):
    """
    Replace the masked region with a Gaussian-blurred version of the input.
        obs  : (C, H, W) float32
        mask : (H, W) in [0, 1] — 1 = fully occlude, 0 = keep original
    The blur is applied across spatial axes only (sigma=0 on the channel axis)
    so both channels of the 2-ch input are treated independently but uniformly.
    """
    blurred = gaussian_filter(obs, sigma=[0, 3, 3])
    return obs * (1.0 - mask) + blurred * mask


def _snapshot_memory(net):
    """Return detached clones of (running_mem, prev_x_conv)."""
    return net.running_mem.clone(), net.prev_x_conv.clone()


def _restore_memory(net, snap):
    """Write a previously snapshotted memory pair back into the model."""
    net.running_mem = snap[0].clone()
    net.prev_x_conv = snap[1].clone()


def compute_sarfa_map(net, obs, logits, action, density, blur_radius, mem_snap, level):
    """
    Compute a (H, W) SARFA saliency map for one frame.

    For each grid cell at stride `density`:
      1. Occlude the observation at that location.
      2. Restore model memory to mem_snap so every probe starts from the same
         temporal state (the state that existed at this frame during rollout).
      3. Query the model for perturbed logits.
      4. Call computeSaliencyUsingSarfa(original_action, q_before, q_after).

    The score grid is bilinearly upsampled to (H, W) and normalised to [0, 1].

    Args:
        mem_snap: Tuple of (running_mem, prev_x_conv) cloned tensors, captured
                  immediately after the real forward pass for this frame.
    """
    C, H, W   = obs.shape
    n_actions = len(logits)
    action_keys   = [str(i) for i in range(n_actions)]
    q_before      = {k: float(logits[int(k)]) for k in action_keys}
    if level == 'a1':
        original_action = str(action)
    else:
        # For a2 saliency, SARFA "action" should be max a2-logit id,
        # while env action remains the a1-level selected action.
        original_action = str(int(np.argmax(logits)))

    rows = list(range(0, H, density))
    cols = list(range(0, W, density))
    score_grid = np.zeros((len(rows), len(cols)), dtype=np.float32)

    hx = cx  = torch.zeros(1)

    for gi, i in enumerate(rows):
        for gj, j in enumerate(cols):
            mask  = _get_mask([i, j], [H, W], blur_radius)
            obs_p = _occlude(obs, mask)
            obs_t = torch.FloatTensor(obs_p).unsqueeze(0)

            # Each probe must start from the same memory state
            _restore_memory(net, mem_snap)
            with contextlib.redirect_stdout(io.StringIO()):
                with torch.no_grad():
                    out = net(obs_t, hx, cx)
            idx = 1 if level == 'a1' else 7
            logits_p = out[idx].squeeze(0).numpy()

            q_after = {k: float(logits_p[int(k)]) for k in action_keys}
            sal, *_ = computeSaliencyUsingSarfa(original_action, q_before, q_after)
            score_grid[gi, gj] = sal

    # Leave model in the pre-scan state; the real rollout already advanced it
    _restore_memory(net, mem_snap)

    # Bilinear upsample from grid resolution → full (H, W)
    saliency = resize(score_grid, (W, H), interpolation=INTER_LINEAR)
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Visualization & Output
# ══════════════════════════════════════════════════════════════════════════════

def overlay_saliency(raw_rgb, saliency_map, alpha=0.5):
    """
    Alpha-blend a jet heatmap over the raw Atari RGB frame.
        raw_rgb      : (H, W, 3) uint8
        saliency_map : (H_s, W_s) float32 in [0, 1]
        alpha        : heatmap opacity (0 = invisible, 1 = fully opaque)
    Returns (H, W, 3) uint8.
    """
    H, W = raw_rgb.shape[:2]
    sal     = resize(saliency_map, (W, H), interpolation=INTER_LINEAR)
    heatmap = (cm.jet(sal)[:, :, :3] * 255).astype(np.float32)
    blended = (1.0 - alpha) * raw_rgb.astype(np.float32) + alpha * heatmap
    return blended.clip(0, 255).astype(np.uint8)


def save_outputs(overlaid_frames, output_dir):
    """
    Write each frame as a PNG to <output_dir>/frames/frame_XXXX.png and
    compile all frames into <output_dir>/saliency.gif at 15 fps.
    """
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    for i, frame in enumerate(overlaid_frames):
        imageio.imwrite(os.path.join(frames_dir, f'frame_{i:04d}.png'), frame)

    gif_path = os.path.join(output_dir, 'saliency.gif')
    imageio.mimsave(gif_path, overlaid_frames, fps=15)

    print(f"[eval_saliency] Saved {len(overlaid_frames)} frames → {frames_dir}")
    print(f"[eval_saliency] GIF  → {gif_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    net, env = load_model_and_env(args)

    # ── Rollout ───────────────────────────────────────────────────────────────
    frames_rgb, obs_list, logits_list, action_list, memory_list = collect_rollout(
        net, env, args.num_frames, args.level
    )

    # ── SARFA scan ────────────────────────────────────────────────────────────
    action_meanings = env.unwrapped.get_action_meanings()
    overlaid_frames = []

    print(f"[eval_saliency] Computing SARFA maps "
          f"(density={args.density}, blur_radius={args.blur_radius})…")

    for step, (raw_rgb, obs, logits, action, mem_snap) in enumerate(
        zip(frames_rgb, obs_list, logits_list, action_list, memory_list)
    ):
        # Use per-frame memory snapshot — correct temporal context for this step
        sal_map  = compute_sarfa_map(
            net, obs, logits, action, args.density, args.blur_radius, mem_snap, args.level
        )
        overlaid = overlay_saliency(raw_rgb, sal_map, alpha=args.alpha)
        overlaid_frames.append(overlaid)

        action_name = (action_meanings[action]
                       if action < len(action_meanings) else str(action))
        print(f"  step {step:04d}  action={action_name:<14s}  max_sal={sal_map.max():.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    save_outputs(overlaid_frames, args.output_dir)


if __name__ == '__main__':
    main()
