"""
Start a training run.

    python src/main.py configs/<config>.ini [parallel_id]

Reads the config, creates one shared model and one shared optimizer, then starts
  - one test process  (test.py):  plays greedily, logs scores, saves checkpoints
  - `workers` training processes (train.py): play and update the shared model
"""
from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import configparser
import sys
import time
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Value

from environment import atari_env
from utils import read_config
import model
from train import train
from test import test
from shared_optim import SharedAdam

config = configparser.ConfigParser()
config.read(sys.argv[1])
c = config['DEFAULT']

args = argparse.Namespace()
# Run
args.experiment_name = c.get('experiment_name')
args.env = c.get('env')
args.env_config = c.get('env_config')
args.seed = c.getint('seed')
args.workers = c.getint('workers')
args.total_steps_stop = c.getint('total_steps_stop')
args.log_dir = c.get('log_dir')
# Environment
args.max_episode_length = c.getint('max_episode_length')
args.skip_rate = c.getint('skip_rate')
args.normalization_alpha = c.getfloat('normalization_alpha')
# Optimization
args.lr = c.getfloat('lr')
args.num_steps = c.getint('num_steps')
args.entropy_coef = c.getfloat('entropy_coef')
# Level 1
args.gamma = c.getfloat('gamma')
args.gamma_oracle = c.getfloat('gamma_oracle')
# Level 2
args.num_options = c.getint('num_options')
args.gamma2 = c.getfloat('gamma2')
args.gamma_oracle2 = c.getfloat('gamma_oracle2')
# Oracle loss weight
args.w_restoration_loss = c.getfloat('w_restoration_loss')
# Monitoring
args.monitor_s_save_interval = c.getint('monitor_s_save_interval')
args.gif_image_save_frequency = c.getint('gif_image_save_frequency')
# Several runs of the same config are told apart by parallel_id (second argument)
args.parallel_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Based on https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training workers update the shared model without locks (Hogwild).

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = model.A3CRules2378OracleTwoLevelMemory(
        env.observation_space.shape[0], env.action_space, args
    )
    shared_model.share_memory()
    frames_total = Value('i', 0)

    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []
    p = mp.Process(target=test, args=(args, shared_model, env_conf, frames_total))
    p.start()
    processes.append(p)
    time.sleep(0.001)
    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf, frames_total))
        p.start()
        processes.append(p)
        time.sleep(0.001)
    for p in processes:
        time.sleep(0.001)
        p.join()
