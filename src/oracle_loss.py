import torch
import torch.nn.functional as F

import model


def compute_oracle_losses(args, player, R, R_intrinsic, w_intrinsic):
    policy_loss = 0
    value_loss = 0
    value_intrinsic_loss = 0
    restoration_loss = 0
    cosine_const_values = []
    gae = torch.zeros_like(R)
    gae_intrinsic = torch.zeros_like(R_intrinsic)

    if args.w_restoration_loss > 0 and len(player.x_restoreds) > 0:
        G_t = player.x_restoreds[-1].clone()
    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        if args.delta_t_mode == 'advantage':
            delta_t = advantage.detach()
        else:
            delta_t = (
                player.rewards[i]
                + args.gamma * player.values[i + 1].data
                - player.values[i].data
            )

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

        if args.delta_t_mode == 'advantage':
            gae = delta_t
            gae_intrinsic = delta_t_intrinsic
        else:
            gae = gae * args.gamma * args.tau + delta_t
            gae_intrinsic = gae_intrinsic * args.gamma * args.tau + delta_t_intrinsic
        policy_advantage = gae + gae_intrinsic
        if args.empirical_distribution_correction:
            policy_advantage = policy_advantage / (player.log_probs[i].detach().exp() + args.empirical_distribution_correction_epsilon)
        policy_loss = (
            policy_loss
            - (player.log_probs[i] * policy_advantage)
            - (args.entropy_coef * player.entropies[i])
        )

    return (
        policy_loss,
        value_loss,
        value_intrinsic_loss,
        restoration_loss,
        cosine_const_values,
    )
