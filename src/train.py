from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
import torch.nn.functional as F
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads, save_checkpoint
import model
from player_util import Agent
import time
import pickle
import csv


def compute_level2_loss_v1(args, player, i, gae2, R2):
    r2_i = player.values[i].detach() * (1 - args.gamma)
    R2 = args.gamma2 * R2 + r2_i
    advantage2 = R2 - player.values2[i]
    value_loss2_i = 0.5 * advantage2.pow(2)

    delta_t2 = (
        r2_i
        + args.gamma2 * player.values2[i + 1].data
        - player.values2[i].data
    )

    gae2 = gae2 * args.gamma2 * args.tau + delta_t2

    return advantage2, value_loss2_i, delta_t2, gae2, R2


def compute_internal_critic_loss(args, player, i, gae_intr, R_intr):
    r_intr = player.values2[i].detach() * (1 - args.gamma)
    R_intr = args.gamma * R_intr + r_intr
    advantage_intr = R_intr - player.values_intr[i]
    value_loss_intr_i = 0.5 * advantage_intr.pow(2)

    delta_intr = (
        r_intr
        + args.gamma * player.values_intr[i + 1].data
        - player.values_intr[i].data
    )

    gae_intr = gae_intr * args.gamma * args.tau + delta_intr

    return advantage_intr, value_loss_intr_i, delta_intr, gae_intr, R_intr


def sampled_action_target(action, logits):
    target = torch.zeros_like(logits)
    return target.scatter(1, action, 1.0)


def train(rank, args, shared_model, optimizer, env_conf, frames_total):
    ptitle(f"Train Agent: {rank}")
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    hidden_size = args.hidden_size
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = model.Hierarchial_interactor_options(
        player.env.observation_space.shape[0], player.env.action_space, args
    )

    player.state = player.env.reset()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = torch.from_numpy(player.state).float().cuda()
            player.model = player.model.cuda()
    else:
        player.state = torch.from_numpy(player.state).float()
    player.model.train()
    if len(args.distributed_step_size) > 0:
        num_steps = args.distributed_step_size[rank % len(args.distributed_step_size)]
    else:
        num_steps = args.num_steps

    game_count = 0
    batch_count = 0
    loss_csv_path = None
    last_save = 0
    milestone_saved = False
    if args.monitor_losses:
        log_dir_path = f"{args.log_dir}{args.experiment_name}/"
        os.makedirs(log_dir_path, exist_ok=True)
        loss_csv_path = f"{log_dir_path}losses_rank{rank}.csv"
        with open(loss_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['batch_num', 'policy_loss', 'value_loss', 'kld_loss', 'restoration_loss'])
    try:
        while 1:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            if player.done:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.cx = torch.zeros(1, hidden_size).cuda()
                        player.hx = torch.zeros(1, hidden_size).cuda()
                else:
                    player.cx = torch.zeros(1, hidden_size)
                    player.hx = torch.zeros(1, hidden_size)
            else:
                player.cx = player.cx.data
                player.hx = player.hx.data
            for step in range(num_steps):
                player.action_train()

                if player.done:
                    break

            if player.done:
                game_count += 1
                if args.monitor_s and (game_count % args.monitor_s_save_interval == 0 or game_count == 1):
                    s_data = {
                        'game': game_count,
                        'rank': rank,
                        's_values': player.model.s_values
                    }
                    log_dir_path = f"{args.log_dir}{args.experiment_name}/"
                    os.makedirs(log_dir_path, exist_ok=True)
                    save_path = f"{log_dir_path}s_monitor_{args.env}_rank{rank}_game{game_count}.pkl"
                    print(f"Saving s monitoring data to: {save_path}")
                    with open(save_path, 'wb') as f:
                        pickle.dump(s_data, f)
                    if os.path.exists(save_path):
                        print(f"S monitoring data saved successfully: {save_path}")
                    else:
                        print(f"ERROR: Failed to save s monitoring data to: {save_path}")
                player.model.s_values = []

                player.eps_len = 0
                state = player.env.reset()
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        player.state = torch.from_numpy(state).float().cuda()
                else:
                    player.state = torch.from_numpy(state).float()

            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    R = torch.zeros(1, 1).cuda()
                    gae = torch.zeros(1, 1).cuda()
                    R2 = torch.zeros(1, 1).cuda()
                    gae2 = torch.zeros(1, 1).cuda()
                    R_intr = torch.zeros(1, 1).cuda()
                    gae_intr = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
                gae = torch.zeros(1, 1)
                R2 = torch.zeros(1, 1)
                gae2 = torch.zeros(1, 1)
                R_intr = torch.zeros(1, 1)
                gae_intr = torch.zeros(1, 1)

            model_output = None
            if not player.done:
                state = player.state

                saved_option = player.model.current_option

                model_output = player.model(
                    state.unsqueeze(0), player.hx, player.cx, None
                )

                player.model.current_option = saved_option

                value = model_output[0]
                R = value.detach()
                value2 = model_output[6]
                R2 = value2.detach()
                value_intr = model_output[12]
                R_intr = value_intr.detach()

            player.values.append(R)
            player.values2.append(R2)
            player.values_intr.append(R_intr)

            bootstrap_a2_logits = None
            bootstrap_interactor_logits = None
            if model_output is not None:
                bootstrap_a2_logits = model_output[7]
                bootstrap_interactor_logits = (
                    model_output[8].detach() + model_output[9]
                )

            policy_loss = 0
            value_loss = 0
            value_loss_intr = 0
            policy_loss2 = 0
            value_loss2 = 0
            interactor_loss = 0
            beta_loss = 0
            interactor_running_target = None
            level2_running_target = None

            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                delta_t = (
                    player.rewards[i]
                    + args.gamma * player.values[i + 1].data
                    - player.values[i].data
                )

                (
                    advantage2,
                    value_loss2_i,
                    delta_t2,
                    gae2,
                    R2
                ) = compute_level2_loss_v1(args, player, i, gae2, R2)

                ### int critic
                value_loss2 = value_loss2 + value_loss2_i

                (
                    _,
                    value_loss_intr_i,
                    _,
                    gae_intr,
                    R_intr
                ) = compute_internal_critic_loss(
                    args, player, i, gae_intr, R_intr
                )
                value_loss_intr = value_loss_intr + value_loss_intr_i

                gae = gae * args.gamma * args.tau + delta_t
                actor1_gae = (gae + gae_intr).detach()
                l2_gae = gae2.detach()

                policy_loss = (
                    policy_loss
                    - (player.log_probs[i] * actor1_gae)
                    - (args.entropy_coef * player.entropies[i])
                )

                beta_adv = l2_gae
                beta_loss = beta_loss + player.betas[i] * beta_adv

                a2_logits_i = player.a2_logits[i]

                if level2_running_target is None:
                    init_a2_logits = (
                        bootstrap_a2_logits
                        if bootstrap_a2_logits is not None
                        else a2_logits_i
                    )
                    level2_running_target = F.softmax(
                        init_a2_logits, dim=1
                    ).detach()

                sampled_target = sampled_action_target(
                    player.actions2[i],
                    a2_logits_i,
                )
                level2_running_target = (
                    args.gamma * level2_running_target
                    + (1 - args.gamma) * sampled_target
                ).detach()

                target_dist = level2_running_target
                pred_log_dist = F.log_softmax(a2_logits_i, dim=1)
                kld_i = F.kl_div(
                    pred_log_dist, target_dist, reduction='batchmean'
                )
                policy_loss2 = policy_loss2 + kld_i * l2_gae
                if args.entropy_coef2 > 0:
                    policy_loss2 = (
                        policy_loss2
                        - (args.entropy_coef2 * player.entropies2[i])
                    )

                a1_logits_i = player.a1_logits[i]
                a_21_logits_i = player.a_21_logits[i]

                if interactor_running_target is None:
                    if bootstrap_interactor_logits is not None:
                        init_interactor_logits = bootstrap_interactor_logits
                    else:
                        init_interactor_logits = (
                            a1_logits_i.detach() + a_21_logits_i
                        )
                    interactor_running_target = F.softmax(
                        init_interactor_logits, dim=1
                    ).detach()

                sampled_target = sampled_action_target(
                    player.actions[i],
                    a_21_logits_i,
                )
                interactor_running_target = (
                    args.gamma * interactor_running_target
                    + (1 - args.gamma) * sampled_target
                ).detach()

                target_dist = interactor_running_target
                pred_log_dist = F.log_softmax(
                    a1_logits_i.detach() + a_21_logits_i, dim=1
                )
                kld_i = F.kl_div(
                    pred_log_dist, target_dist, reduction='batchmean'
                )
                interactor_loss = interactor_loss + kld_i * l2_gae

            ### total loss value
            value_loss = value_loss + value_loss2 + value_loss_intr
            beta_term = args.beta_coef * beta_loss
            total_loss = (
                policy_loss + policy_loss2 + 0.5 * value_loss
                + interactor_loss + beta_term
            )

            player.model.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(player.model.parameters(), 40.0)
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()

            if args.save_model_steps > 0 and frames_total.value // args.save_model_steps > last_save and rank == 0:
                last_save = frames_total.value // args.save_model_steps
                log_dir_path = f"{args.log_dir}{args.experiment_name}/"
                os.makedirs(log_dir_path, exist_ok=True)
                torch.save(shared_model.state_dict(), f"{log_dir_path}model_{frames_total.value}.dat")

            if args.monitor_losses:
                batch_count += 1
                with open(loss_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        batch_count,
                        policy_loss.item(),
                        (0.5 * value_loss).item(),
                        0,
                        0,
                    ])

            player.clear_actions()
            steps_taken = step + 1 if player.done else num_steps
            frames_total.value += steps_taken
            if (
                rank == 0
                and not milestone_saved
                and args.save_model_milestone_steps > 0
                and frames_total.value >= args.save_model_milestone_steps
            ):
                milestone_saved = True
                milestone_path = save_checkpoint(
                    shared_model.state_dict(), args, 'milestone', frames_total.value
                )
                print(f"Saved milestone checkpoint: {milestone_path}")
            if frames_total.value > args.total_steps_stop:
                break
    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print(f"train agent {rank} process finished")
