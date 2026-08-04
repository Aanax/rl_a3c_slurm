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


def running_return_td(R, reward, value, gamma):
    """One step of discounted return and TD advantage (level-agnostic).

    Updates the running return R <- gamma * R + reward, then forms
    advantage = R - value. Returns (R, advantage, delta) where delta is
    advantage.detach() for use as an actor/beta weight.
    """
    R = gamma * R + reward
    advantage = R - value
    return R, advantage, advantage.detach()


def compute_level2_loss_v1(args, player, i, R2):
    r2_i = player.values[i].detach() * (1 - args.gamma)
    R2, advantage2, delta2 = running_return_td(
        R2, r2_i, player.values2[i], args.gamma2
    )
    value_loss2_i = 0.5 * advantage2.pow(2)
    return advantage2, value_loss2_i, delta2, R2


def sampled_action_target(action, logits):
    target = torch.zeros_like(logits)
    return target.scatter(1, action, 1.0)


def _action_equal(a, b):
    return int(a.item()) == int(b.item())


def _should_update_beta(actions, i, n):
    """True if beta at step i should get a gradient.

    Update when the option continues into i+1, or on the last step of a
    multi-step option (same as i-1, different from i+1). Skip length-1
    options and the final rollout step (no i+1 to observe).
    """
    if i + 1 >= n:
        return False
    if _action_equal(actions[i], actions[i + 1]):
        return True
    return i > 0 and _action_equal(actions[i], actions[i - 1])


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
    player.model = getattr(model, args.model_type)(
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
    use_beta = getattr(args, 'use_beta_termination', True)
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
                    R2 = torch.zeros(1, 1).cuda()
            else:
                R = torch.zeros(1, 1)
                R2 = torch.zeros(1, 1)

            model_output = None
            if not player.done:
                state = player.state

                model_output = player.model(
                    state.unsqueeze(0), player.hx, player.cx, None,
                    bootstrap_only=True
                )

                # HierarchialLevelsOutput has named fields; legacy models use
                # the same layout for V1 / V2 / a2_logits at indices 0 / 6 / 7.
                if isinstance(model_output, model.HierarchialLevelsOutput):
                    R = model_output.V1.detach()
                    R2 = model_output.V2.detach()
                    bootstrap_a2_logits = model_output.a2_logits
                else:
                    R = model_output[0].detach()
                    R2 = model_output[6].detach()
                    bootstrap_a2_logits = model_output[7]
            else:
                bootstrap_a2_logits = None

            player.values.append(R)
            player.values2.append(R2)

            policy_loss = 0
            value_loss = 0
            policy_loss2 = 0
            value_loss2 = 0
            beta_loss = 0
            last_idx = len(player.rewards) - 1
            if bootstrap_a2_logits is not None:
                init_a2_logits = bootstrap_a2_logits
            else:
                init_a2_logits = player.a2_logits[last_idx]
            level2_running_target = F.softmax(
                init_a2_logits, dim=1
            ).detach()

            n_rewards = len(player.rewards)
            has_beta1 = use_beta and len(player.betas1) == n_rewards
            has_beta2 = use_beta and len(player.betas2) == n_rewards

            for i in reversed(range(n_rewards)):
                R, advantage, delta = running_return_td(
                    R, player.rewards[i], player.values[i], args.gamma
                )
                value_loss = value_loss + 0.5 * advantage.pow(2)

                (
                    advantage2,
                    value_loss2_i,
                    delta2,
                    R2
                ) = compute_level2_loss_v1(args, player, i, R2)

                value_loss2 = value_loss2 + value_loss2_i

                actor_delta = delta + delta2

                policy_loss = (
                    policy_loss
                    - (player.log_probs[i] * actor_delta)
                    - (args.entropy_coef * player.entropies[i])
                )

                if has_beta1 and _should_update_beta(
                    player.actions, i, n_rewards
                ):
                    beta_loss = beta_loss + player.betas1[i] * delta
                if has_beta2 and _should_update_beta(
                    player.actions2, i, n_rewards
                ):
                    beta_loss = beta_loss + player.betas2[i] * delta2

                a2_logits_i = player.a2_logits[i]

                sampled_target = sampled_action_target(
                    player.actions2[i],
                    a2_logits_i,
                )
                level2_running_target = (
                    args.gamma2 * level2_running_target
                    + (1 - args.gamma2) * sampled_target
                ).detach()

                target_prob = level2_running_target
                pred_log_prob = F.log_softmax(a2_logits_i, dim=1)
                ce_i = -(target_prob * pred_log_prob).sum(dim=1)
                policy_loss2 = policy_loss2 + ce_i * actor_delta
                if args.entropy_coef2 > 0:
                    policy_loss2 = (
                        policy_loss2
                        - (args.entropy_coef2 * player.entropies2[i])
                    )

            value_loss = value_loss + value_loss2
            beta_term = args.beta_coef * beta_loss
            total_loss = (
                policy_loss + policy_loss2 + 0.5 * value_loss
                + beta_term
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
