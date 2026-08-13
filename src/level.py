from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


LevelOut = namedtuple(
    'LevelOut',
    ['V', 'logits', 'action_idx', 'action_onehot', 'beta_grad', 'terminated'],
)


def _init_linear(linear, weight_scale=1.0):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    std = 1.0 / math.sqrt(fan_in)
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    linear.bias.data.fill_(0)
    if weight_scale != 1.0:
        linear.weight.data.mul_(weight_scale)


class Level(nn.Module):
    """One hierarchical level: encoder + actor + critic + beta + sticky action.

    Args:
        encoder: feature extractor for this level.
        feat_dim: flattened size of encoder features.
        n_actions: number of discrete actions/options at this level.
        upper_options_dim: size of the higher-level option one-hot concatenated
            onto features before the actor/critic/beta heads. 0 means this
            level has no upper option context (e.g. the top level).
    """

    def __init__(
        self,
        encoder,
        feat_dim,
        n_actions,
        upper_options_dim=0,
        actor_weight_scale=0.01,
        critic_weight_scale=1.0,
        beta_weight_scale=0.01,
    ):
        super(Level, self).__init__()
        self.encoder = encoder
        self.feat_dim = feat_dim
        self.n_actions = n_actions
        self.upper_options_dim = upper_options_dim
        in_dim = feat_dim + upper_options_dim

        self.actor = nn.Linear(in_dim, n_actions)
        self.critic = nn.Linear(in_dim, 1)
        self.beta = nn.Linear(in_dim, n_actions)

        _init_linear(self.actor, weight_scale=actor_weight_scale)
        _init_linear(self.critic, weight_scale=critic_weight_scale)
        _init_linear(self.beta, weight_scale=beta_weight_scale)

        self.current_action = None
        self.last_beta_logits = None

    def encode(self, x):
        features, _, _, _ = self.encoder(x)
        return features

    def _head_input(self, features, upper_options_onehot=None):
        """Build the vector fed to actor/critic/beta heads.

        If upper_options_dim > 0, concatenates the higher-level option one-hot
        onto flattened features.
        """
        flat = features.view(features.size(0), -1)
        if self.upper_options_dim > 0:
            if upper_options_onehot is None:
                raise ValueError(
                    'upper_options_onehot required when upper_options_dim > 0'
                )
            return torch.cat([flat, upper_options_onehot], dim=1)
        return flat

    def forward_heads(
        self,
        features,
        upper_options_onehot=None,
        bootstrap_only=False,
        force_terminate=False,
    ):
        """Run actor/critic/beta and sticky action sampling for this level.

        Args:
            features: encoder output for this level.
            upper_options_onehot: optional one-hot of the active higher-level
                option (length upper_options_dim). Ignored when
                upper_options_dim == 0.
            bootstrap_only: if True, do not update current_action / monitoring.
            force_terminate: if True, skip beta sampling and resample from the
                policy (higher-level terminate, or 1-step actions when
                gamma_actor=0).
        """
        head_in = self._head_input(features, upper_options_onehot)
        logits = self.actor(head_in)
        V = self.critic(head_in)
        beta_logits = self.beta(head_in)
        beta = torch.sigmoid(beta_logits)

        if not bootstrap_only:
            self.last_beta_logits = beta_logits.detach()

        probs = F.softmax(logits, dim=1)
        prev_action = self.current_action

        if bootstrap_only and prev_action is not None and prev_action.size(0) == probs.size(0):
            action_idx = prev_action.argmax(dim=1, keepdim=True)
            terminated = False
        elif (
            force_terminate
            or prev_action is None
            or prev_action.size(0) != probs.size(0)
        ):
            # Fresh choice (episode start / higher-level terminate): treat as
            # terminated so training updates the policy, not beta.
            action_idx = probs.multinomial(1)
            terminated = True
        else:
            beta_active = (beta.detach() * prev_action).sum(dim=1, keepdim=True)
            beta_active = beta_active.clamp(0.0, 1.0)
            beta_active = torch.where(
                torch.isfinite(beta_active),
                beta_active,
                torch.full_like(beta_active, 0.5),
            )
            terminate = torch.bernoulli(beta_active)
            prev_idx = prev_action.argmax(dim=1, keepdim=True)
            new_idx = probs.multinomial(1)
            action_idx = torch.where(terminate.bool(), new_idx, prev_idx)
            terminated = terminate.bool().item()

        action_onehot = torch.zeros_like(probs)
        action_onehot.scatter_(1, action_idx, 1.0)
        if not bootstrap_only:
            self.current_action = action_onehot.detach()

        # beta of the action chosen at this step; train updates beta when the
        # sampled terminate flag is 0, and the policy when it is 1.
        beta_grad = (beta * action_onehot).sum(dim=1, keepdim=True)

        return LevelOut(
            V=V,
            logits=logits,
            action_idx=action_idx,
            action_onehot=action_onehot,
            beta_grad=beta_grad,
            terminated=terminated,
        )

    def reset_action(self):
        self.current_action = None
