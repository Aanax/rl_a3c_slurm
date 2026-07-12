from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import configparser
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Value
from environment import atari_env
from utils import read_config
import model
from train import train
from test import test
from shared_optim import SharedRMSprop, SharedAdam
#import gym.configuration import undo_logger_setup
import time
import sys
import random

# Config file reading setup
config = configparser.ConfigParser()
config.read(sys.argv[1]) #'./configs/run_config.ini'

# Create args namespace and populate from config
args = argparse.Namespace()
args.lr = config.getfloat('DEFAULT', 'lr', fallback=0.0001)
args.entropy_coef = config.getfloat('DEFAULT', 'entropy_coef', fallback=0.01)
args.entropy_coef2 = config.getfloat('DEFAULT', 'entropy_coef2', fallback=0.0)
args.value_coef = config.getfloat('DEFAULT', 'value_coef', fallback=0.5)
args.gamma = config.getfloat('DEFAULT', 'gamma', fallback=0.99)
args.gamma_memory = config.getfloat('DEFAULT', 'gamma_memory', fallback=0.99)
args.gamma2 = config.getfloat('DEFAULT', 'gamma2', fallback=0.99)
args.tau = config.getfloat('DEFAULT', 'tau', fallback=1.00)
args.seed = config.getint('DEFAULT', 'seed', fallback=1)
args.workers = config.getint('DEFAULT', 'workers', fallback=32)
args.num_steps = config.getint('DEFAULT', 'num_steps', fallback=20)
args.max_episode_length = config.getint('DEFAULT', 'max_episode_length', fallback=10000)
args.env = config.get('DEFAULT', 'env', fallback='PongNoFrameSkip-v4')
args.shared_optimizer = config.getboolean('DEFAULT', 'shared_optimizer', fallback=True)
args.load = config.getboolean('DEFAULT', 'load', fallback=False)
args.save_max = config.getboolean('DEFAULT', 'save_max', fallback=False)
args.optimizer = config.get('DEFAULT', 'optimizer', fallback='Adam')
args.load_model_dir = config.get('DEFAULT', 'load_model_dir', fallback='trained_models/')
args.save_model_dir = config.get('DEFAULT', 'save_model_dir', fallback='trained_models/')
args.log_dir = config.get('DEFAULT', 'log_dir', fallback='logs/')
args.amsgrad = config.getboolean('DEFAULT', 'amsgrad', fallback=False)
args.skip_rate = config.getint('DEFAULT', 'skip_rate', fallback=4)
args.hidden_size = config.getint('DEFAULT', 'hidden_size', fallback=512)
args.tensorboard_logger = config.getboolean('DEFAULT', 'tensorboard_logger', fallback=False)
args.gif_image_save_frequency = config.getint('DEFAULT', 'gif_image_save_frequency', fallback=100000)
args.env_config = config.get('DEFAULT', 'env_config', fallback='configs/envs_config.json')
args.experiment_name = config.get('DEFAULT', 'experiment_name', fallback='unnamed_experiment')
args.total_steps_stop = config.getint('DEFAULT', 'total_steps_stop', fallback=100000000)
args.input_normalization_class = config.get('DEFAULT', 'input_normalization_class', fallback='NormalizedEnv')
args.normalization_alpha = config.getfloat('DEFAULT', 'normalization_alpha', fallback=0.9999)
args.model_type = config.get('DEFAULT', 'model_type', fallback='Hierarchial_interactor_options')
args.num_options = config.getint('DEFAULT', 'num_options', fallback=8)
args.monitor_s = config.getboolean('DEFAULT', 'monitor_s', fallback=False)
args.monitor_s_save_interval = config.getint('DEFAULT', 'monitor_s_save_interval', fallback=500)
args.w_kld_loss = config.getfloat('DEFAULT', 'w_kld_loss', fallback=0.0)
args.w_restoration_loss = config.getfloat('DEFAULT', 'w_restoration_loss', fallback=0.0)
args.monitor_losses = config.getboolean('DEFAULT', 'monitor_losses', fallback=False)
args.save_model_steps = config.getint('DEFAULT', 'save_model_steps', fallback=0)
args.save_model_milestone_steps = config.getint(
    'DEFAULT', 'save_model_milestone_steps', fallback=2000000
)
args.use_beta_termination = config.getboolean('DEFAULT', 'use_beta_termination', fallback=True)
args.beta_coef = config.getfloat('DEFAULT', 'beta_coef', fallback=1.0)
args.train_version = config.get('DEFAULT', 'train_version', fallback='v1')

# For list types
gpu_ids_str = config.get('DEFAULT', 'gpu_ids', fallback='-1')
args.gpu_ids = [int(x.strip()) for x in gpu_ids_str.split(',')] if gpu_ids_str else [-1]

dist_str = config.get('DEFAULT', 'distributed_step_size', fallback='')
args.distributed_step_size = [int(x.strip()) for x in dist_str.split(',')] if dist_str else []

# Parse experiment_name and parallel_id from argv

try:
    args.parallel_id = int(sys.argv[2])
except:
    args.parallel_id = 0
# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

if __name__ == '__main__':
    if args.seed == 0:
        args.seed = random.randint(0, 1000000)
    torch.manual_seed(args.seed)
    if args.gpu_ids != [-1]:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    env = atari_env(args.env, env_conf, args)
    shared_model = getattr(model, args.model_type)(env.observation_space.shape[0], env.action_space, args)
    if args.load:
        saved_state = torch.load(
            f"{args.load_model_dir}{args.env}.dat",
            map_location=lambda storage, loc: storage,
        )
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()
    frames_total = Value('i', 0)

    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

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
