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



def _option_index(option_id):
    if torch.is_tensor(option_id):
        return int(option_id.item())
    return int(option_id)


def activation_bootstrap(index, n_steps, batch_tail, plugs):
    """Value after the last execution of an option in the batch.

    Args:
        index: Step where the option executes.
        n_steps: Number of steps in the batch.
        batch_tail: Value at the state after the batch.
        plugs: Value of this option at the next state, one entry per step.

    Returns:
        Detached bootstrap.
    """
    if index + 1 >= n_steps:
        return batch_tail.detach()
    plug = plugs[index]
    if plug is None:
        raise ValueError(
            'missing bootstrap for an option that ends before the batch tail'
        )
    return plug.detach()


def option_activation_returns(active, returns, option_id, reward, gamma, bootstrap):
    """Update one backward step of every option seen so far.

    Args:
        active: Mutable flags, one per option.
        returns: Mutable targets, one per option.
        option_id: Option executing at this step.
        reward: Reward added to the executing option.
        gamma: Discount.
        bootstrap: Tail used the first time ``option_id`` is seen.

    Returns:
        Target of the executing option.
    """
    for option_k, is_active in enumerate(active):
        if option_k == option_id:
            if not is_active:
                if bootstrap is None:
                    raise ValueError('activation bootstrap is missing')
                returns[option_k] = bootstrap.detach()
                active[option_k] = True
            returns[option_k] = gamma * returns[option_k] + reward
        elif is_active:
            returns[option_k] = gamma * returns[option_k]
    return returns[option_id]


def level2_batch_return_step(running_return, reward, gamma):
    """One backward step of the level-2 return across the batch.

    Args:
        running_return: Return accumulated from later steps.
        reward: Level-2 reward at this step.
        gamma: Level-2 discount.

    Returns:
        Discounted return at this step.
    """
    return gamma * running_return + reward


def oracle_option_target(
    future_target, delta, gamma, option_ended, bootstrap=None,
):
    """Discounted state-change target inside one option.

    Args:
        future_target: Target accumulated after this step.
        delta: State change at this step.
        gamma: Discount of this oracle target.
        option_ended: True when this step ends the option.
        bootstrap: Oracle prediction for the option that just ended.
            Zero target when omitted.

    Returns:
        Gamma times the future target plus delta. A finished option uses
        ``bootstrap`` as that future target.
    """
    if option_ended:
        if bootstrap is None:
            return torch.zeros_like(delta)
        future_target = bootstrap
    return gamma * future_target.detach() + delta


def oracle_two_level_actor_loss(
    log_prob1, advantage_int, entropy1,
    log_prob2, advantage2, entropy2, entropy_coef,
):
    """Actor losses for the oracle two-level model.

    Args:
        log_prob1: Log-probability of the environment action.
        advantage_int: Detached internal-critic advantage.
        entropy1: Entropy of the level-1 policy.
        log_prob2: Log-probability of the sampled option.
        advantage2: Detached advantage of the level-2 policy.
        entropy2: Entropy of the option policy.
        entropy_coef: Coefficient of the entropy bonus.

    Returns:
        Tuple of level-1 and level-2 actor loss terms for one step.
    """
    policy1 = -(log_prob1 * advantage_int) - entropy_coef * entropy1
    policy2 = -(log_prob2 * advantage2) - entropy_coef * entropy2
    return policy1, policy2


def external_advantage_restoration(cosine_loss, advantage_ext, weight):
    """Scale the oracle cosine loss by the external advantage.

    Args:
        cosine_loss: Negative cosine between the decoder output and the target.
        advantage_ext: Detached external-critic advantage.
        weight: Restoration loss weight.

    Returns:
        Weighted restoration term for one step.
    """
    return weight * cosine_loss * advantage_ext


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
                    R_intrinsic2 = torch.zeros(1, 1).cuda()
                    gae2 = torch.zeros(1, 1).cuda()
                    gae_intrinsic = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
                R_intrinsic = torch.zeros(1, 1)
                gae = torch.zeros(1, 1)
                R2 = torch.zeros(1, 1)
                R_intrinsic2 = torch.zeros(1, 1)
                gae2 = torch.zeros(1, 1)
                gae_intrinsic = torch.zeros(1, 1)
            model_output = None
            oracle_bootstrap = None
            oracle2_bootstrap = None
            w_intrinsic = 0.0
            if isinstance(player.model, (
                model.A3CRules2378OracleIntrinsicCritic,
                model.A3CRules2378OracleFCIntrinsicCritic,
                model._IntrinsicCriticMixin,
            )):
                w_intrinsic = 1.0
            if not player.done:
                state = player.state
                if isinstance(player.model, model.A3CRules2378OracleTwoLevel) and player.actions2:
                    model_output = player.model(
                        state.unsqueeze(0),
                        player.hx,
                        player.cx,
                        option_index=player.actions2[-1],
                    )
                else:
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
                if isinstance(player.model, model.A3CRules2378OracleTwoLevel):
                    oracle_bootstrap = model_output[4].detach()
                    oracle2_bootstrap = model_output[-1].detach()
                    player.s1_states.append(model_output[-3])
                    player.s2_states.append(model_output[-2])
                    R_intrinsic2 = model_output[-4].detach()
            player.values.append(R)
            player.values_intrinsic.append(R_intrinsic)
            # Check if model is hierarchical (has V2 and a2 outputs)
            # If values2 was populated during action_train, model is hierarchical
            is_hierarchical = len(player.values2) > 0
            use_oracle_two_level = isinstance(
                player.model, model.A3CRules2378OracleTwoLevel
            )
            if is_hierarchical:
                # Append final R2 to match the final R we just appended
                # If episode is done, R2 remains zeros (bootstrap value)
                player.values2.append(R2)
            if use_oracle_two_level:
                player.values_intrinsic2.append(R_intrinsic2)
            policy_loss = 0
            value_loss = 0
            value_intrinsic_loss = 0
            value_intrinsic2_loss = 0
            restoration_loss = 0
            policy_loss2 = 0
            value_loss2 = 0
            cosine_const_values = []
            
            if args.w_restoration_loss > 0 and len(player.x_restoreds) > 0:
                if oracle_bootstrap is None:
                    G_t = torch.zeros_like(player.x_restoreds[-1])
                else:
                    G_t = oracle_bootstrap
            if len(player.rewards) > 0:
                if len(player.s1_states) != len(player.rewards) + 1:
                    raise ValueError('s1 transitions are misaligned')
                if len(player.s2_states) != len(player.rewards) + 1:
                    raise ValueError('s2 transitions are misaligned')
            if args.w_restoration_loss > 0 and len(player.oracle2_preds) > 0:
                if oracle2_bootstrap is None:
                    G2 = torch.zeros_like(player.oracle2_preds[-1])
                else:
                    G2 = oracle2_bootstrap
            if len(player.rewards) > 0:
                num_options = player.model.num_options
                active_ext = [False] * num_options
                returns_ext = [None] * num_options
                active_int = [False] * num_options
                returns_int = [None] * num_options
                external_batch_tail = player.values[-1].detach()
                intrinsic_batch_tail = player.values_intrinsic[-1].detach()
                R2 = player.values2[-1].detach()
            for i in reversed(range(len(player.rewards))):
                option_id = _option_index(player.actions2[i])
                n_steps = len(player.rewards)
                external_tail = None
                if not active_ext[option_id]:
                    external_tail = activation_bootstrap(
                        i, n_steps, external_batch_tail,
                        player.option_end_values,
                    )
                R1 = option_activation_returns(
                    active_ext, returns_ext, option_id,
                    player.rewards[i], args.gamma, external_tail,
                )
                advantage = R1 - player.values[i]
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

                R2 = args.gamma2 * R2 + (1.0 - args.gamma) * R1
                advantage2 = R2 - player.values2[i]
                delta2 = advantage2.detach()
                value_loss2 = value_loss2 + 0.5 * advantage2.pow(2)

                # Intrinsic critic target (oracle reward).
                if args.w_restoration_loss > 0 and len(player.x_restoreds) > 0:
                    episode_ended = player.done and i + 1 == len(player.rewards)
                    option_ended = (
                        i + 1 < n_steps
                        and option_id != _option_index(player.actions2[i + 1])
                    )
                    oracle_plug = None
                    if option_ended:
                        oracle_plug = player.option_end_oracle[i]
                    G_t = oracle_option_target(
                        G_t,
                        player.s1_states[i + 1] - player.s1_states[i],
                        args.gamma_oracle,
                        option_ended or episode_ended,
                        oracle_plug,
                    )

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

                    intrinsic_tail = None
                    if not active_int[option_id]:
                        intrinsic_tail = activation_bootstrap(
                            i, n_steps, intrinsic_batch_tail,
                            player.option_end_intrinsic,
                        )
                    R_intrinsic = option_activation_returns(
                        active_int, returns_int, option_id,
                        oracle_r, args.gamma, intrinsic_tail,
                    )
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
                    restoration_loss = restoration_loss + external_advantage_restoration(
                        cosine_restoreds, delta_t + delta2, args.w_restoration_loss
                    )
                else:
                    delta_t_intrinsic = 0.0

                advantage_intrinsic2 = torch.zeros_like(advantage2)
                if args.w_restoration_loss > 0 and len(player.oracle2_preds) > i:
                    batch_ended = player.done and i + 1 == len(player.rewards)
                    G2 = oracle_option_target(
                        G2,
                        player.s2_states[i + 1] - player.s2_states[i],
                        args.gamma_oracle2,
                        batch_ended,
                    )
                    cosine2 = -F.cosine_similarity(
                        player.oracle2_preds[i].view(-1),
                        G2.detach().view(-1),
                        dim=0,
                    )
                    restoration_loss = restoration_loss + external_advantage_restoration(
                        cosine2, delta2, args.w_restoration_loss,
                    )
                    cosine_const2 = F.cosine_similarity(
                        player.oracle2_preds[i].detach().view(-1),
                        G2.detach().view(-1),
                        dim=0,
                    ).squeeze(0)
                    oracle_r2 = (1.0 - args.gamma2) * cosine_const2
                    R_intrinsic2 = level2_batch_return_step(
                        R_intrinsic2, oracle_r2, args.gamma2,
                    )
                    advantage_intrinsic2 = R_intrinsic2 - player.values_intrinsic2[i]
                    value_intrinsic2_loss = (
                        value_intrinsic2_loss + 0.5 * advantage_intrinsic2.pow(2)
                    )

                if args.delta_t_mode == 'advantage':
                    gae = delta_t
                    gae_intrinsic = delta_t_intrinsic
                elif args.delta_t_mode == 'td':
                    gae = gae * args.gamma * args.tau + delta_t
                    gae_intrinsic = gae_intrinsic * args.gamma * args.tau + delta_t_intrinsic
                policy1_i, policy2_i = oracle_two_level_actor_loss(
                    player.log_probs[i],
                    gae_intrinsic,
                    player.entropies[i],
                    player.log_probs2[i],
                    advantage_intrinsic2.detach(),
                    player.entropies2[i],
                    args.entropy_coef,
                )
                policy_loss = policy_loss + policy1_i
                policy_loss2 = policy_loss2 + policy2_i

            kld_loss = 0
            if args.w_kld_loss > 0:
                for i in range(len(player.rewards)):
                    if len(player.kls) > i:
                        kld_loss += args.w_kld_loss * player.kls[i]

            value_loss = value_loss + value_loss2
            value_intrinsic_loss = value_intrinsic_loss + value_intrinsic2_loss
            total_loss = (
                policy_loss
                + policy_loss2
                + 0.5 * value_loss
                + 0.5 * value_intrinsic_loss
                + kld_loss
                + restoration_loss
            )
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
