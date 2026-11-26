"""Small shared tools: logging, config reading, checkpoints, gradient sharing."""
from __future__ import division
import glob
import json
import logging
import os

import torch


def setup_logger(logger_name, log_file, level=logging.INFO):
    """Log both to a file and to the screen."""
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read a JSON config."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_checkpoint(state_dict, args, kind, steps, score):
    """Save model weights and delete the previous file of the same kind.

    The file goes to ``{log_dir}{experiment_name}/models/p{parallel_id}/`` and is
    named like ``best_steps1200_score21.dat`` (a score of 2.5 is written ``2p50``).

    Args:
        state_dict: Model weights.
        args: Run config.
        kind: ``latest`` or ``best``.
        steps: frames_total at save time.
        score: Game score.

    Returns:
        Path of the saved file.
    """
    save_dir = os.path.join(args.log_dir, args.experiment_name, "models", f"p{args.parallel_id}")
    os.makedirs(save_dir, exist_ok=True)
    for old_path in glob.glob(os.path.join(save_dir, f"{kind}_*.dat")):
        os.remove(old_path)
    if score == int(score):
        score_text = str(int(score))
    else:
        score_text = f"{score:.2f}".replace('.', 'p')
    path = os.path.join(save_dir, f"{kind}_steps{steps}_score{score_text}.dat")
    torch.save(state_dict, path)
    return path


def ensure_shared_grads(model, shared_model):
    """Point the shared model's gradients at this worker's gradients.

    Each worker process has its own view of the shared model, so this link is
    per worker. It is made once: after that the shared gradients already point
    at this worker's gradient tensors, so the loop stops early. That is why
    train.py zeroes gradients in place (zero_grad(set_to_none=False)).
    """
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
