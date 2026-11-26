"""
One training worker. Many of these run in parallel (A3C) and share one model.

Each worker repeats:
  1. copy the shared model
  2. play num_steps steps (a batch), or fewer if a life is lost
  3. walk the batch backwards in time and compute the losses
  4. send the gradients to the shared model and make one optimizer step

What each part of the model learns from:
  Level 1 external critic   game rewards, per option (see update_option_returns)
  Level 2 external critic   level-1 returns, scaled by (1 - gamma)
  Oracle 1                  the change of s1, inside the current option
  Oracle 2                  the change of s2
  Level 1 intrinsic critic  oracle-1 reward = (1 - gamma) * cosine(oracle 1 guess, its target)
  Level 2 intrinsic critic  oracle-2 reward = (1 - gamma2) * cosine(oracle 2 guess, its target)
  Level 1 actor             level-1 intrinsic advantage
  Level 2 actor             level-2 intrinsic advantage
  Oracle losses are weighted by the external advantages.
"""
from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
from setproctitle import setproctitle as ptitle
import torch
import torch.nn.functional as F

from environment import atari_env
from utils import ensure_shared_grads
import model
from player_util import Agent
import time
import pickle
import csv


def update_option_returns(returns, playing_option, reward, gamma, start_value):
    """One step backwards in time for the returns of every option.

    Each option keeps its own return. Going backwards through the batch:
      - the option that played at this step gets   R = gamma * R + reward
      - every other option already met gets         R = gamma * R
      - an option met for the first time starts from `start_value`
        (its value at the state where it stopped playing)

    Args:
        returns: List with one return per option. None = not met yet. Changed in place.
        playing_option: Option that played at this step.
        reward: Reward of this step.
        gamma: Discount.
        start_value: Starting return of `playing_option` if it is met for the first time.

    Returns:
        The return of the playing option.
    """
    if returns[playing_option] is None:
        returns[playing_option] = start_value.detach()
    for option in range(len(returns)):
        if returns[option] is None:
            continue
        if option == playing_option:
            returns[option] = gamma * returns[option] + reward
        else:
            returns[option] = gamma * returns[option]
    return returns[playing_option]


def train(rank, args, shared_model, optimizer, env_conf, frames_total):
    ptitle(f"Train Agent: {rank}")
    torch.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.model = model.A3CRules2378OracleTwoLevelMemory(
        player.env.observation_space.shape[0], player.env.action_space, args
    )
    player.state = torch.from_numpy(player.env.reset()).float()
    player.model.train()

    log_dir_path = f"{args.log_dir}{args.experiment_name}/"
    os.makedirs(log_dir_path, exist_ok=True)
    loss_csv_path = f"{log_dir_path}losses_rank{rank}.csv"
    with open(loss_csv_path, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(
            ['batch_num', 'policy_loss', 'value_loss', 'restoration_loss', 'value_intrinsic_loss']
        )
    cosine_csv_path = f"{log_dir_path}cosine_const_rank{rank}.csv"
    with open(cosine_csv_path, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(['batch_num', 'step_idx', 'cosine_const'])

    game_count = 0
    batch_count = 0
    try:
        while True:
            player.model.load_state_dict(shared_model.state_dict())

            # ---------------------------------------------------------------
            # 1. Play one batch
            # ---------------------------------------------------------------
            for step in range(args.num_steps):
                player.action_train()
                if player.done:
                    break
            n_steps = len(player.rewards)

            # ---------------------------------------------------------------
            # 2. Values at the state after the batch (the "tail")
            #    After a lost life there is no next state: everything is zero.
            # ---------------------------------------------------------------
            if player.done:
                value_tail = torch.zeros(1, 1)
                value_intrinsic_tail = torch.zeros(1, 1)
                value2_tail = torch.zeros(1, 1)
                value_intrinsic2_tail = torch.zeros(1, 1)
                oracle1_target = torch.zeros_like(player.oracle1_predictions[-1])
                oracle2_target = torch.zeros_like(player.oracle2_predictions[-1])
            else:
                # Keep the option that played last: it has not ended yet.
                tail = player.model(
                    player.state.unsqueeze(0), player.previous_s1, option=player.options[-1]
                )
                value_tail = tail['value'].detach()
                value_intrinsic_tail = tail['value_intrinsic'].detach()
                value2_tail = tail['value2'].detach()
                value_intrinsic2_tail = tail['value_intrinsic2'].detach()
                oracle1_target = tail['oracle1_prediction'].detach()
                oracle2_target = tail['oracle2_prediction'].detach()
                player.s1_states.append(tail['s1'].detach())
                player.s2_states.append(tail['s2'])

            # ---------------------------------------------------------------
            # 3. Walk the batch backwards and add up the losses
            # ---------------------------------------------------------------
            num_options = player.model.num_options
            returns_external = [None] * num_options     # level 1, one return per option
            returns_intrinsic = [None] * num_options    # level 1, one return per option
            return2 = value2_tail                       # level 2
            return_intrinsic2 = value_intrinsic2_tail   # level 2

            policy_loss = 0
            policy_loss2 = 0
            value_loss = 0
            value_loss2 = 0
            value_intrinsic_loss = 0
            value_intrinsic2_loss = 0
            restoration_loss = 0
            cosine_const_values = []

            for i in reversed(range(n_steps)):
                option = player.options[i].item()
                is_last_step = (i == n_steps - 1)
                episode_ended_here = player.done and is_last_step
                option_ended_here = (not is_last_step) and player.options[i + 1].item() != option

                # --- Level 1 external critic -------------------------------
                # An option met for the first time starts from the value at the state
                # where it stopped: the batch tail, or the value stored when it ended.
                if is_last_step:
                    start_value = value_tail
                else:
                    start_value = player.option_end_values[i]
                return1 = update_option_returns(
                    returns_external, option, player.rewards[i], args.gamma, start_value
                )
                advantage1 = return1 - player.values[i]
                value_loss = value_loss + 0.5 * advantage1.pow(2)

                # --- Level 2 external critic -------------------------------
                # Its reward is the level-1 return, scaled by (1 - gamma).
                return2 = args.gamma2 * return2 + (1.0 - args.gamma) * return1
                advantage2 = return2 - player.values2[i]
                value_loss2 = value_loss2 + 0.5 * advantage2.pow(2)

                # --- Oracle 1 ----------------------------------------------
                # Target: discounted change of s1 until the end of the current option.
                # When the option ends here, the future part comes from the guess of the
                # finished option at the state where it ended.
                # When a life is lost here, the target is zero.
                s1_change = player.s1_states[i + 1] - player.s1_states[i]
                if episode_ended_here:
                    oracle1_target = torch.zeros_like(s1_change)
                else:
                    if option_ended_here:
                        oracle1_target = player.option_end_oracle1[i]
                    oracle1_target = args.gamma_oracle * oracle1_target.detach() + s1_change
                oracle1_guess = player.oracle1_predictions[i].view(-1)
                oracle1_target_flat = oracle1_target.detach().view(-1)

                cosine1 = F.cosine_similarity(oracle1_guess, oracle1_target_flat, dim=0)
                restoration_loss = restoration_loss + (
                    args.w_restoration_loss * (-cosine1) * (advantage1.detach() + advantage2.detach())
                )

                # --- Level 1 intrinsic critic ------------------------------
                cosine1_const = cosine1.detach()
                cosine_const_values.append(cosine1_const.item())
                oracle1_reward = (1.0 - args.gamma) * cosine1_const
                if is_last_step:
                    start_value = value_intrinsic_tail
                else:
                    start_value = player.option_end_values_intrinsic[i]
                return_intrinsic1 = update_option_returns(
                    returns_intrinsic, option, oracle1_reward, args.gamma, start_value
                )
                advantage_intrinsic1 = return_intrinsic1 - player.values_intrinsic[i]
                value_intrinsic_loss = value_intrinsic_loss + 0.5 * advantage_intrinsic1.pow(2)

                # --- Oracle 2 ----------------------------------------------
                # Target: discounted change of s2 (options do not matter here).
                s2_change = player.s2_states[i + 1] - player.s2_states[i]
                if episode_ended_here:
                    oracle2_target = torch.zeros_like(s2_change)
                else:
                    oracle2_target = args.gamma_oracle2 * oracle2_target.detach() + s2_change
                oracle2_guess = player.oracle2_predictions[i].view(-1)
                oracle2_target_flat = oracle2_target.detach().view(-1)

                cosine2 = F.cosine_similarity(oracle2_guess, oracle2_target_flat, dim=0)
                restoration_loss = restoration_loss + (
                    args.w_restoration_loss * (-cosine2) * advantage2.detach()
                )

                # --- Level 2 intrinsic critic ------------------------------
                oracle2_reward = (1.0 - args.gamma2) * cosine2.detach()
                return_intrinsic2 = args.gamma2 * return_intrinsic2 + oracle2_reward
                advantage_intrinsic2 = return_intrinsic2 - player.values_intrinsic2[i]
                value_intrinsic2_loss = value_intrinsic2_loss + 0.5 * advantage_intrinsic2.pow(2)

                # --- Actors ------------------------------------------------
                # Both actors follow their level's intrinsic advantage.
                policy_loss = (
                    policy_loss
                    - player.log_probs[i] * advantage_intrinsic1.detach()
                    - args.entropy_coef * player.entropies[i]
                )
                policy_loss2 = (
                    policy_loss2
                    - player.log_probs2[i] * advantage_intrinsic2.detach()
                    - args.entropy_coef * player.entropies2[i]
                )

            # ---------------------------------------------------------------
            # 4. Update the shared model
            # ---------------------------------------------------------------
            total_loss = (
                policy_loss
                + policy_loss2
                + 0.5 * (value_loss + value_loss2)
                + 0.5 * (value_intrinsic_loss + value_intrinsic2_loss)
                + restoration_loss
            )
            # Zero in place: the shared model keeps pointing at these gradient tensors.
            player.model.zero_grad(set_to_none=False)
            total_loss.backward()
            ensure_shared_grads(player.model, shared_model)
            optimizer.step()

            # The memory keeps gradients inside one batch only.
            if player.previous_s1 is not None:
                player.previous_s1 = player.previous_s1.detach()

            # ---------------------------------------------------------------
            # 5. Logs
            # ---------------------------------------------------------------
            batch_count += 1
            with open(loss_csv_path, 'a', newline='') as csvfile:
                csv.writer(csvfile).writerow([
                    batch_count,
                    policy_loss.item(),
                    (0.5 * (value_loss + value_loss2)).item(),
                    restoration_loss.item(),
                    (0.5 * (value_intrinsic_loss + value_intrinsic2_loss)).item(),
                ])
            with open(cosine_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for step_idx, cosine_const in enumerate(cosine_const_values):
                    writer.writerow([batch_count, step_idx, cosine_const])

            # ---------------------------------------------------------------
            # 6. After a lost life: save s1 monitoring and reset the game
            # ---------------------------------------------------------------
            if player.done:
                game_count += 1
                if game_count % args.monitor_s_save_interval == 0 or game_count == 1:
                    save_path = f"{log_dir_path}s_monitor_{args.env}_rank{rank}_game{game_count}.pkl"
                    with open(save_path, 'wb') as f:
                        pickle.dump(
                            {'game': game_count, 'rank': rank, 's_values': player.s1_history}, f
                        )
                    print(f"S monitoring data saved: {save_path}")
                player.s1_history = []
                player.eps_len = 0
                player.state = torch.from_numpy(player.env.reset()).float()

            player.clear_actions()
            frames_total.value += n_steps
            if frames_total.value > args.total_steps_stop:
                break
    except KeyboardInterrupt:
        time.sleep(0.01)
        print("KeyboardInterrupt exception is caught")
    finally:
        print(f"train agent {rank} process finished")
