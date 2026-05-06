from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
import torch.nn.functional as F
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
import model
from player_util import Agent
from torch.autograd import Variable
import time
import pickle
import csv
import os


def compute_level2_loss_v1(args, player, i, gae2, R2):
    """
    v1: orig algo for level 2 loss.
    
    r2_i = V1_i * (1 - gamma1)
    R2 = gamma2 * R2 + r2_i
    advantage2 = R2 - V2[i]
    """
    # r2_i = V1_i (r for critic2 is V1)
    r2_i = player.values[i].detach() * (1 - args.gamma)
    R2 = args.gamma2 * R2 + r2_i
    advantage2 = R2 - player.values2[i]
    value_loss2_i = 0.5 * advantage2.pow(2)
    
    # Generalized Advantage Estimation for level 2
    delta_t2 = (
        r2_i
        + args.gamma2 * player.values2[i + 1].data
        - player.values2[i].data
    )
    
    gae2 = gae2 * args.gamma2 * args.tau + delta_t2
    
    return advantage2, value_loss2_i, delta_t2, gae2, R2


def compute_level2_loss_a2a1_connect(args, player, i, gae2, R2, logprobs_summ):
    """
    v1: orig algo for level 2 loss.
    
    r2_i = V1_i * (1 - gamma1)
    R2 = gamma2 * R2 + r2_i
    advantage2 = R2 - V2[i]
    """
    # r2_i = V1_i (r for critic2 is V1)
    r2_i = player.values[i].detach() * (1 - args.gamma)
    R2 = args.gamma2 * R2 + r2_i
    advantage2 = R2 - player.values2[i]
    value_loss2_i = 0.5 * advantage2.pow(2)


    logprobs_summ = logprobs_summ*args.gamma + player.log_probs2[i]*(1-args.gamma)
    
    # Generalized Advantage Estimation for level 2
    delta_t2 = (
        r2_i
        + args.gamma2 * player.values2[i + 1].data
        - player.values2[i].data
    )
    
    gae2 = gae2 * args.gamma2 * args.tau + delta_t2
    
    return advantage2, value_loss2_i, delta_t2, gae2, R2, logprobs_summ


def compute_level2_loss_v2(args, player, i, r2, V2Target, gae2):
    """
    v2: new algo for level 2 loss.
    
    r2 := g1 * r2 + (1 - g1) * r[i]
    V2Target := V2Target * g2 + r2
    advantage2 := V2Target.detach - V2[i]
    """
    # r2 = g1 * r2 + (1 - g1) * r[i]
    r2 = args.gamma * r2 + (1 - args.gamma) * player.rewards[i]
    # V2Target := V2Target * g2 + r2
    V2Target = V2Target * args.gamma2 + r2
    # a2 = V2Target.detach - V2[i]
    advantage2 = V2Target.detach() - player.values2[i]
    value_loss2_i = 0.5 * advantage2.pow(2)
    
    # Use V1(1-g1) for r2_i and GAE for level 2
    r2_i = player.values[i].detach() * (1 - args.gamma)
    delta_t2 = (
        r2_i
        + args.gamma2 * player.values2[i + 1].data
        - player.values2[i].data
    )
    gae2 = gae2 * args.gamma2 * args.tau + delta_t2
    
    return advantage2, value_loss2_i, delta_t2, gae2, r2, V2Target


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
    player.model = getattr(model, args.model_type)(player.env.observation_space.shape[0], player.env.action_space, args)

    player.state = player.env.reset()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = torch.from_numpy(player.state).float().cuda()
            player.model = player.model.cuda()
    else:
        player.state = torch.from_numpy(player.state).float()
    player.model.train()
    if len(args.distributed_step_size) > 0:
        num_steps = args.distributed_step_size[rank%len(args.distributed_step_size)]
    else:
        num_steps = args.num_steps

    game_count = 0
    batch_count = 0
    loss_csv_path = None
    last_save = 0
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
                    # Save s values for this game
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
                    # Check if file was saved successfully
                    if os.path.exists(save_path):
                        print(f"S monitoring data saved successfully: {save_path}")
                    else:
                        print(f"ERROR: Failed to save s monitoring data to: {save_path}")
                    # Clear s values for next games
                player.model.s_values = []

                # # Reset memory for models with memory when starting new episode
                # if hasattr(player.model, 'reset_memory'):
                #     player.model.reset_memory()

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
                    last_log_prob2 = torch.zeros(1, 1).cuda()

            else:
                R = torch.zeros(1, 1)
                gae = torch.zeros(1, 1)
                R2 = torch.zeros(1, 1)
                gae2 = torch.zeros(1, 1)
                last_log_prob2 = torch.zeros(1, 1)
                
                
            model_output = None
            if not player.done:
                state = player.state

                if hasattr(args, 'model_type') and args.model_type == 'Hierarchial_memory_action_memrelu':
                    model_output = player.model(
                        state.unsqueeze(0), player.hx, player.cx, None, player.action_prev
                    )
                else:
                    model_output = player.model(
                        state.unsqueeze(0), player.hx, player.cx, None
                    )

                value = model_output[0]
                R = value.detach()
                # For hierarchical models, also get V2
                if len(model_output) >= 8:
                    value2 = model_output[6]
                    R2 = value2.detach()
                    logit2 = model_output[7]
                    prob2 = F.softmax(logit2, dim=1)
                    last_log_prob2 = F.log_softmax(logit2, dim=1)
                    action2 = prob2.multinomial(1).data # note: may sample differently in reality
                    last_log_prob2 = last_log_prob2.gather(1, action2).detach()

            player.values.append(R)
            
            # Check if model is hierarchical (has V2 and a2 outputs)
            # If values2 was populated during action_train, model is hierarchical
            is_hierarchical = len(player.values2) > 0
            if is_hierarchical:
                # Append final R2 to match the final R we just appended
                # If episode is done, R2 remains zeros (bootstrap value)
                player.values2.append(R2)
            if args.use_train_a2a1_connect:
                player.log_probs2.append(last_log_prob2)

            policy_loss = 0
            value_loss = 0
            policy_loss2 = 0
            value_loss2 = 0
            logprobs_summ = player.log_probs2[-1]
            
            # Determine which train version to use for level 2 calculations
            train_version = getattr(args, 'train_version', 'v1')
            use_train_v2 = (train_version == 'v2')
            
            # trainv2: init r2 and V2Target at the start of the batch
            if is_hierarchical and use_train_v2:
                # r2 = (1-g1)*V1[H].detach() - w last value (botstrap)
                r2 = (1 - args.gamma) * player.values[-1].detach()
                # V2Target = V2[H].detach() - use the last value2 (bootstrap)
                V2Target = player.values2[-1].detach()
            
            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion 1
                delta_t = (
                    player.rewards[i]
                    + args.gamma * player.values[i + 1].data
                    - player.values[i].data
                )

                # Level 2 loss
                delta_t2 = None
                if is_hierarchical and len(player.values2) > i and len(player.log_probs2) > i:
                    if args.use_train_a2a1_connect:
                        (
                            advantage2,
                            value_loss2_i,
                            delta_t2,
                            gae2,
                            R2,
                            logprobs_summ
                        ) = compute_level2_loss_a2a1_connect(
                            args, player, i, gae2, R2, logprobs_summ
                        )
                        value_loss2 = value_loss2 + value_loss2_i

                    elif use_train_v2:
                        (
                            advantage2,
                            value_loss2_i,
                            delta_t2,
                            gae2,
                            r2,
                            V2Target
                        ) = compute_level2_loss_v2(
                            args, player, i, r2, V2Target, gae2
                        )
                        value_loss2 = value_loss2 + value_loss2_i
                    else:
                        (
                            advantage2,
                            value_loss2_i,
                            delta_t2,
                            gae2,
                            R2
                        ) = compute_level2_loss_v1(
                            args, player, i, gae2, R2
                        )
                        value_loss2 = value_loss2 + value_loss2_i

                # For actor1: use only delta_t when separate_actor_deltas=True,
                # otherwise combine delta_t + delta_t2 (original behaviour).
                separate_deltas = getattr(args, 'separate_actor_deltas', False)
                if delta_t2 is not None and not separate_deltas:
                    gae = gae * args.gamma * args.tau + (delta_t + delta_t2)
                else:
                    gae = gae * args.gamma * args.tau + delta_t
                    
                policy_loss = (
                    policy_loss
                    - (player.log_probs[i] * gae)
                    - (args.entropy_coef * player.entropies[i])
                )
                
                # Actor2 loss (only for hierarchical)
                if args.use_train_a2a1_connect:
                    policy_loss2 = (
                        policy_loss2
                        + 0.5*((player.log_probs2[i] - logprobs_summ)**2) * gae
                    )
                elif is_hierarchical and len(player.log_probs2) > i:
                    policy_loss2 = (
                        policy_loss2
                        - (player.log_probs2[i] * gae2)
                        - (args.entropy_coef * player.entropies2[i])
                    )
                

            # Additional losses for VAE models (only compute if weights > 0)
            kld_loss = 0
            restoration_loss = 0
            if args.w_kld_loss > 0 or args.w_restoration_loss > 0:
                batch_size = len(player.rewards)
                for i in range(len(player.rewards)):
                    if args.w_kld_loss > 0 and len(player.kls) > i:
                        kld_loss += args.w_kld_loss * player.kls[i]
                    if args.w_restoration_loss > 0 and len(player.x_restoreds) > i:
                        restoration_loss += args.w_restoration_loss * (player.x_restoreds[i] - player.states[i].detach()).pow(2).mean()

            # Combine critic1 loss with critic2 loss
            if is_hierarchical:
                value_loss = value_loss + value_loss2
            
            # Total loss: actor1 + actor2 + combined critic loss
            if is_hierarchical:
                total_loss = policy_loss + policy_loss2 + 0.5 * value_loss + kld_loss + restoration_loss
            else:
                total_loss = policy_loss + 0.5 * value_loss + kld_loss + restoration_loss

            player.model.zero_grad()
            total_loss.backward()
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()

            # Reset memory for models with memory when starting new batch
            if hasattr(player.model, 'reset_memory'):
                player.model.reset_memory()

            if hasattr(shared_model, 'orthogonalize_conv4'):
                shared_model.orthogonalize_conv4()

            if args.save_model_steps > 0 and frames_total.value // args.save_model_steps > last_save and rank == 0:
                last_save = frames_total.value // args.save_model_steps
                log_dir_path = f"{args.log_dir}{args.experiment_name}/"
                os.makedirs(log_dir_path, exist_ok=True)
                torch.save(shared_model.state_dict(), f"{log_dir_path}model_{frames_total.value}.dat")

            # Save losses to CSV if monitoring is enabled
            if args.monitor_losses:
                batch_count += 1
                with open(loss_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        batch_count,
                        policy_loss.item(),
                        (0.5 * value_loss).item(),
                        kld_loss.item() if isinstance(kld_loss, torch.Tensor) else kld_loss,
                        restoration_loss.item() if isinstance(restoration_loss, torch.Tensor) else restoration_loss
                    ])

            player.clear_actions()
            steps_taken = step + 1 if player.done else num_steps
            frames_total.value += steps_taken
            if frames_total.value > args.total_steps_stop:
                break
    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print(f"train agent {rank} process finished")
