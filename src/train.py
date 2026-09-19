from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
import torch.nn.functional as F

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
    cosine_csv_path = None
    last_save = 0
    if args.monitor_losses:
        log_dir_path = f"{args.log_dir}{args.experiment_name}/"
        os.makedirs(log_dir_path, exist_ok=True)
        loss_csv_path = f"{log_dir_path}losses_rank{rank}.csv"
        with open(loss_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['batch_num', 'policy_loss', 'value_loss', 'kld_loss', 'restoration_loss', 'value_intrinsic_loss'])
    if args.monitor_cosine_const:
        log_dir_path = f"{args.log_dir}{args.experiment_name}/"
        os.makedirs(log_dir_path, exist_ok=True)
        cosine_csv_path = f"{log_dir_path}cosine_const_rank{rank}.csv"
        with open(cosine_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['batch_num', 'step_idx', 'cosine_const'])
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
                    R_intrinsic = torch.zeros(1, 1).cuda()
                    gae = torch.zeros(1, 1).cuda()
                    R2 = torch.zeros(1, 1).cuda()
                    gae2 = torch.zeros(1, 1).cuda()
                    gae_intrinsic = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
                R_intrinsic = torch.zeros(1, 1)
                gae = torch.zeros(1, 1)
                R2 = torch.zeros(1, 1)
                gae2 = torch.zeros(1, 1)
                gae_intrinsic = torch.zeros(1, 1)
            model_output = None
            if not player.done:
                state = player.state
                model_output = player.model(
                    state.unsqueeze(0), player.hx, player.cx
                )
                if isinstance(player.model, model._FutureSharedDiffTargetMixin):
                    player.shared_states.append(model_output[-1].detach())
                value = model_output[0]
                R = value.detach()
                if isinstance(player.model, (model.A3CRules2378OracleIntrinsicCritic, model.A3CRules2378OracleFCIntrinsicCritic, model._IntrinsicCriticMixin)) and len(model_output) >= 6:
                    R_intrinsic = model_output[5].detach()
                    w_intrinsic = 1.0
                else:
                    R_intrinsic = value.detach()
                    w_intrinsic = 0.0
                # For hierarchical models, also get V2
                if len(model_output) >= 8:
                    value2 = model_output[6]
                    R2 = value2.detach()
            player.values.append(R)
            player.values_intrinsic.append(R_intrinsic)
            # Check if model is hierarchical (has V2 and a2 outputs)
            # If values2 was populated during action_train, model is hierarchical
            is_hierarchical = len(player.values2) > 0
            if is_hierarchical:
                # Append final R2 to match the final R we just appended
                # If episode is done, R2 remains zeros (bootstrap value)
                player.values2.append(R2)
            policy_loss = 0
            value_loss = 0
            value_intrinsic_loss = 0
            restoration_loss = 0
            policy_loss2 = 0
            value_loss2 = 0
            cosine_const_values = []
            
            # Determine which train version to use for level 2 calculations
            train_version = getattr(args, 'train_version', 'v1')
            use_train_v2 = (train_version == 'v2')
            
            # trainv2: init r2 and V2Target at the start of the batch
            if is_hierarchical and use_train_v2:
                # r2 = (1-g1)*V1[H].detach() - w last value (botstrap)
                r2 = (1 - args.gamma) * player.values[-1].detach()
                # V2Target = V2[H].detach() - use the last value2 (bootstrap)
                V2Target = player.values2[-1].detach()

            if args.w_restoration_loss > 0 and len(player.x_restoreds) > 0:
                G_t = player.x_restoreds[-1].clone()
            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion 1
                if args.delta_t_mode == 'advantage':
                    delta_t = advantage.detach()
                else:
                    delta_t = (
                        player.rewards[i]
                        + args.gamma * player.values[i + 1].data
                        - player.values[i].data
                    )

                # Intrinsic critic target (oracle reward).
                if args.w_restoration_loss > 0 and len(player.x_restoreds) > 0:
                    if isinstance(player.model, model._FutureSharedDiffTargetMixin):
                        if len(player.shared_states) != len(player.rewards) + 1:
                            raise ValueError('shared feature transitions are misaligned')
                        G_t = (G_t * args.gamma_restoration
                               + player.shared_states[i + 1] - player.shared_states[i])
                    else:
                        G_t = G_t * args.gamma_restoration + player.next_states[i] - player.states[i]

                    if getattr(player.model, 'predicts_shared_diff', False):
                        G_const = G_t.detach().view(-1)
                    elif args.relu_g_const:
                        G_const = F.relu(G_t.detach().view(-1))
                    else:
                        G_const = G_t.detach().view(-1)

                    cosine_const = F.cosine_similarity(
                        player.x_restoreds[i].detach().view(-1),
                        G_const,
                        dim=0
                    ).squeeze(0)
                    cosine_const_values.append(cosine_const.item())
                    oracle_r = (1.0 - args.gamma) * cosine_const

                    R_intrinsic = args.gamma * R_intrinsic + oracle_r
                    intrinsic_advantage = R_intrinsic - player.values_intrinsic[i]
                    value_intrinsic_loss = value_intrinsic_loss + 0.5 * intrinsic_advantage.pow(2)

                    if args.delta_t_mode == 'advantage':
                        delta_t_intrinsic = intrinsic_advantage.detach() * w_intrinsic
                    else:
                        delta_t_intrinsic = (
                            oracle_r
                            + args.gamma * player.values_intrinsic[i + 1].data
                            - player.values_intrinsic[i].data
                        ) * w_intrinsic

                    cosine_restoreds = -F.cosine_similarity(
                        player.x_restoreds[i].view(-1),
                        G_const,
                        dim=0,
                    )
                    restoration_loss = restoration_loss + args.w_restoration_loss * cosine_restoreds * delta_t
                else:
                    delta_t_intrinsic = 0.0

                # Level 2 loss
                delta_t2 = None
                if is_hierarchical and len(player.values2) > i and len(player.log_probs2) > i:
                    if use_train_v2:
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

                # For actor1, use sum of delta_t and delta_t2 (if hierarchical)
                if args.delta_t_mode == 'advantage':
                    if delta_t2 is not None:
                        raise ValueError('delta_t_mode=advantage does not support hierarchical delta_t2')
                    gae = delta_t
                    gae_intrinsic = delta_t_intrinsic
                elif delta_t2 is not None:
                    gae = gae * args.gamma * args.tau + (delta_t + delta_t2)
                else:
                    gae = gae * args.gamma * args.tau + delta_t

                if args.delta_t_mode == 'td':
                    gae_intrinsic = gae_intrinsic * args.gamma * args.tau + delta_t_intrinsic
                policy_advantage = gae + gae_intrinsic
                if args.empirical_distribution_correction:
                    policy_advantage = policy_advantage / (player.log_probs[i].detach().exp() + args.empirical_distribution_correction_epsilon)
                policy_loss = (
                    policy_loss
                    - (player.log_probs[i] * policy_advantage)
                    - (args.entropy_coef * player.entropies[i])
                )
                
                # Actor2 loss (only for hierarchical)
                if is_hierarchical and len(player.log_probs2) > i:
                    policy_loss2 = (
                        policy_loss2
                        - (player.log_probs2[i] * gae2)
                        - (args.entropy_coef * player.entropies2[i])
                    )

            # Additional losses for VAE models (only compute if weights > 0)
            kld_loss = 0
            if args.w_kld_loss > 0:
                batch_size = len(player.rewards)
                for i in range(len(player.rewards)):
                    if args.w_kld_loss > 0 and len(player.kls) > i:
                        kld_loss += args.w_kld_loss * player.kls[i]

            # Combine critic1 loss with critic2 loss
            if is_hierarchical:
                value_loss = value_loss + value_loss2
            
            # Total loss: actor1 + actor2 + combined critic loss
            if is_hierarchical:
                total_loss = policy_loss + policy_loss2 + 0.5 * value_loss + kld_loss + restoration_loss
            else:
                total_loss = policy_loss + 0.5 * value_loss + kld_loss + restoration_loss + 0.5 * value_intrinsic_loss * w_intrinsic
            player.model.zero_grad()
            total_loss.backward()
            ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
            optimizer.step()

            # Detach prev_shared at the batch boundary: the model keeps the graph in
            # prev_shared for within-rollout BPTT (see _SharedFeatureDiffMixin); detaching it
            # here stops the next batch's first diff from backpropagating into this freed graph.
            if getattr(player.model, 'prev_shared', None) is not None:
                player.model.prev_shared = player.model.prev_shared.detach()
            if getattr(player.model, 'memdiff_sum', None) is not None:
                player.model.memdiff_sum = player.model.memdiff_sum.detach()

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
                        restoration_loss.item() if isinstance(restoration_loss, torch.Tensor) else restoration_loss,
                        (0.5 * value_intrinsic_loss).item() if isinstance(value_intrinsic_loss, torch.Tensor) else 0.5 * value_intrinsic_loss,
                    ])
            elif args.monitor_cosine_const:
                batch_count += 1

            if args.monitor_cosine_const:
                with open(cosine_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for step_idx, cosine_const in enumerate(cosine_const_values):
                        writer.writerow([batch_count, step_idx, cosine_const])

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
