from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple

from level import Level


# Named fields for Hierarchial_levels.forward (also indexable as a plain tuple).
# Legacy slot names hx/cx/mem/x_restored kept for A3C call-site compatibility;
# they are unused placeholders in this model.
HierarchialLevelsOutput = namedtuple(
    'HierarchialLevelsOutput',
    [
        'V1',            # 0  level-1 critic value
        'a1_logits',     # 1  level-1 action logits
        'hx',            # 2  unused LSTM placeholder
        'cx',            # 3  unused LSTM placeholder
        'mem',           # 4  unused
        'x_restored',    # 5  unused
        'V2',            # 6  level-2 critic value
        'a2_logits',     # 7  level-2 option logits
        'a1',            # 8  sampled/persisted level-1 action index
        'a2',            # 9  sampled/persisted level-2 option index
        'beta1',         # 10 level-1 termination coeff (active action)
        'beta2',         # 11 level-2 termination coeff (active option)
        'terminated1',   # 12 whether level-1 action terminated this step
        'terminated2',   # 13 whether level-2 option terminated this step
    ],
)


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class EncoderRules234(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64, use_rmsnorm=False):
        super(EncoderRules234, self).__init__()
        self.use_rmsnorm = use_rmsnorm
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, latent_dim_conv, 3, stride=1, padding=1)
        self.rmsnorm = RMSNorm((1024))
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            fan = fan_in if conv is self.conv1 else (fan_in + fan_out) / 2
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        if self.use_rmsnorm:
            x = x.view(x.size(0), -1)
            x = self.rmsnorm(x)
        return x, None, None, None


class EncoderRules234_2(nn.Module):
    def __init__(self):
        super(EncoderRules234_2, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1.weight)
        if self.conv1.bias is not None:
            self.conv1.bias.data.fill_(0)
        if fan_in > 0:
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std
            nn.init.uniform_(self.conv1.weight, -bound, bound)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x, None, None, None


class Hierarchial_interactor_options(nn.Module):
    """DEPRECATED: superseded by Hierarchial_concat_options.

    Kept only for reproducibility of past runs / comparison. Do not use for
    new experiments: the interactor + internal critic (V_intr) design was
    found to be theoretically unjustified (see epic/001_remove_interactor_and_internal_critic.md).
    New experiments should use Hierarchial_concat_options instead.
    """
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_interactor_options, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n
        self.num_outputs = num_outputs
        self.num_options = getattr(args, 'num_options', 8)

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)
        self.level2_encoder = EncoderRules234_2()

        self.critic_linear2 = nn.Linear(32*4*4, 1)
        self.actor_linear2 = nn.Linear(32*4*4, self.num_options)
        self.beta_linear = nn.Linear(32*4*4, self.num_options)

        self.interactor = nn.Linear(self.num_options, num_outputs)

        self.critic_linear = nn.Linear(64*4*4, 1)
        self.critic_linear_intr = nn.Linear(64*4*4, 1)
        self.actor_linear = nn.Linear(64*4*4, num_outputs)

        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        for linear in [self.critic_linear, self.critic_linear_intr, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)
        self.critic_linear_intr.weight.data.mul_(1.0)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.interactor.weight)
        std = 0.01 / math.sqrt(fan_in)
        nn.init.normal_(self.interactor.weight, mean=0.0, std=std)
        self.interactor.bias.data.fill_(0)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.beta_linear.weight)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.beta_linear.weight, mean=0.0, std=std)
        self.beta_linear.weight.data.mul_(0.01)
        self.beta_linear.bias.data.fill_(0.0)

        self.current_option = None
        self.last_beta_logits = None

        self.train()

    def forward(self, inputs, hx, cx, mem=None, bootstrap_only=False):
        """bootstrap_only: mock forward for the last out-of-batch call in train
        that is only used to read off V1/V2/V_intr. Must have no side effects:
        does not sample/terminate options, does not touch self.current_option,
        and does not write to monitoring buffers.
        """
        s, _, _, _ = self.level1_encoder(inputs)

        if self.monitor_s and not bootstrap_only:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        s2 = s
        s2, _, _, _ = self.level2_encoder(s2)
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)
        beta_logits = self.beta_linear(s2_flat)
        beta = torch.sigmoid(beta_logits)
        if not bootstrap_only:
            self.last_beta_logits = beta_logits.detach()

        if getattr(self, 'monitor_beta', False) and not bootstrap_only:
            if not hasattr(self, 'beta_values'):
                self.beta_values = []
            self.beta_values.append(beta.detach().cpu())
            if not hasattr(self, 'beta_logits_values'):
                self.beta_logits_values = []
            self.beta_logits_values.append(beta_logits.detach().cpu())

        a2_probs = F.softmax(a2_logits, dim=1)

        prev_option = self.current_option

        if bootstrap_only and prev_option is not None and prev_option.size(0) == a2_probs.size(0):
            a2_sample = prev_option.argmax(dim=1, keepdim=True)
            option_terminated = False
        elif prev_option is None or prev_option.size(0) != a2_probs.size(0):
            a2_sample = a2_probs.multinomial(1)
            option_terminated = False
        else:
            beta_active = (beta.detach() * prev_option).sum(dim=1, keepdim=True)
            beta_active = beta_active.clamp(0.0, 1.0)
            beta_active = torch.where(
                torch.isfinite(beta_active),
                beta_active,
                torch.full_like(beta_active, 0.5),
            )
            terminate = torch.bernoulli(beta_active)
            prev_idx = prev_option.argmax(dim=1, keepdim=True)
            new_idx = a2_probs.multinomial(1)
            a2_sample = torch.where(terminate.bool(), new_idx, prev_idx)
            option_terminated = terminate.bool().item()

        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)
        if not bootstrap_only:
            self.current_option = a2_onehot.detach()

        if prev_option is not None and prev_option.size(0) == a2_probs.size(0):
            beta_active_grad = (beta * prev_option).sum(dim=1, keepdim=True)
        else:
            beta_active_grad = torch.zeros(
                beta.size(0), 1, device=beta.device, dtype=beta.dtype
            ) #TODO how grds compute

        a_21_logits = self.interactor(a2_onehot)

        s_flat = s.view(s.size(0), -1)
        # s_flat = F.relu(s_flat)
        a1_logits = self.actor_linear(s_flat)
        V1 = self.critic_linear(s_flat)
        V_intr = self.critic_linear_intr(s_flat)

        combined_logits = a1_logits + a_21_logits.detach()

        return (V1, combined_logits, hx, cx, None, None, V2, a2_logits,
                a1_logits, a_21_logits, a2_sample, beta_active_grad, V_intr,
                option_terminated)


class Hierarchial_concat_options(nn.Module):
    """Options hierarchy without an interactor and without an internal critic.

    Level 2: s2 -> a2 (option policy), s2 -> beta (termination), s2 -> V2.
    Level 1: concat(s1, a2_onehot) -> a1 (action policy), and the SAME
             concat(s1, a2_onehot) -> V1 (critic). a2's sampled one-hot is
             fed as plain (non-differentiable) context to both the level-1
             actor and the level-1 critic, instead of being combined with
             a1 through a separate interactor network.

    There is no internal critic (V_intr): the level-1 actor is trained on
    the sum of its own critic's TD-error (delta) and the level-2 critic's
    TD-error (delta2), computed directly -- not through a proxy critic that
    treats V2 as a pseudo-reward for a downstream critic head.
    """

    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_concat_options, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n
        self.num_outputs = num_outputs
        self.num_options = getattr(args, 'num_options', 8)

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)
        self.level2_encoder = EncoderRules234_2()

        self.critic_linear2 = nn.Linear(32*4*4, 1)
        self.actor_linear2 = nn.Linear(32*4*4, self.num_options)
        self.beta_linear = nn.Linear(32*4*4, self.num_options)

        actor1_critic1_in = 64*4*4 + self.num_options
        self.critic_linear = nn.Linear(actor1_critic1_in, 1)
        self.actor_linear = nn.Linear(actor1_critic1_in, num_outputs)

        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.beta_linear.weight)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.beta_linear.weight, mean=0.0, std=std)
        self.beta_linear.weight.data.mul_(0.01)
        self.beta_linear.bias.data.fill_(0.0)

        self.current_option = None
        self.last_beta_logits = None

        self.train()

    def forward(self, inputs, hx, cx, mem=None, bootstrap_only=False):
        """bootstrap_only: mock forward for the last out-of-batch call in train
        that is only used to read off V1/V2. Must have no side effects:
        does not sample/terminate options, does not touch self.current_option,
        and does not write to monitoring buffers.
        """
        s, _, _, _ = self.level1_encoder(inputs)

        if self.monitor_s and not bootstrap_only:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        s2 = s
        s2, _, _, _ = self.level2_encoder(s2)
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)
        beta_logits = self.beta_linear(s2_flat)
        beta = torch.sigmoid(beta_logits)
        if not bootstrap_only:
            self.last_beta_logits = beta_logits.detach()

        if getattr(self, 'monitor_beta', False) and not bootstrap_only:
            if not hasattr(self, 'beta_values'):
                self.beta_values = []
            self.beta_values.append(beta.detach().cpu())
            if not hasattr(self, 'beta_logits_values'):
                self.beta_logits_values = []
            self.beta_logits_values.append(beta_logits.detach().cpu())

        a2_probs = F.softmax(a2_logits, dim=1)

        prev_option = self.current_option

        if bootstrap_only and prev_option is not None and prev_option.size(0) == a2_probs.size(0):
            a2_sample = prev_option.argmax(dim=1, keepdim=True)
            option_terminated = False
        elif prev_option is None or prev_option.size(0) != a2_probs.size(0):
            a2_sample = a2_probs.multinomial(1)
            option_terminated = False
        else:
            beta_active = (beta.detach() * prev_option).sum(dim=1, keepdim=True)
            beta_active = beta_active.clamp(0.0, 1.0)
            beta_active = torch.where(
                torch.isfinite(beta_active),
                beta_active,
                torch.full_like(beta_active, 0.5),
            )
            terminate = torch.bernoulli(beta_active)
            prev_idx = prev_option.argmax(dim=1, keepdim=True)
            new_idx = a2_probs.multinomial(1)
            a2_sample = torch.where(terminate.bool(), new_idx, prev_idx)
            option_terminated = terminate.bool().item()

        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)
        if not bootstrap_only:
            self.current_option = a2_onehot.detach()

        if prev_option is not None and prev_option.size(0) == a2_probs.size(0):
            beta_active_grad = (beta * prev_option).sum(dim=1, keepdim=True)
        else:
            beta_active_grad = torch.zeros(
                beta.size(0), 1, device=beta.device, dtype=beta.dtype
            )

        s_flat = s.view(s.size(0), -1)
        # a2_onehot carries no gradient back to actor_linear2/beta_linear (it is
        # constructed via scatter_ with a constant 1.0), so this concatenation
        # only feeds a2 as context into actor1/critic1, it does not leak
        # actor1/critic1 gradients back into the level-2 heads.
        actor1_critic1_in = torch.cat([s_flat, a2_onehot], dim=1)

        a1_logits = self.actor_linear(actor1_critic1_in)
        V1 = self.critic_linear(actor1_critic1_in)

        return (V1, a1_logits, hx, cx, None, None, V2, a2_logits,
                a2_sample, beta_active_grad, option_terminated)


class Hierarchial_interactor_options_zeroing_(nn.Module):
    """Options variant with level-2 actor and interactor zeroed; only level-1 actor plays."""

    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_interactor_options_zeroing, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n
        self.num_outputs = num_outputs
        self.num_options = 8

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)
        self.level2_encoder = EncoderRules234_2()

        self.critic_linear2 = nn.Linear(32*4*4, 1)
        self.actor_linear2 = nn.Linear(32*4*4, self.num_options)
        self.beta_linear = nn.Linear(32*4*4, self.num_options)

        self.interactor = nn.Linear(self.num_options, num_outputs)

        self.critic_linear = nn.Linear(64*4*4, 1)
        self.actor_linear = nn.Linear(64*4*4, num_outputs)

        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.interactor.weight)
        std = 0.01 / math.sqrt(fan_in)
        nn.init.normal_(self.interactor.weight, mean=0.0, std=std)
        self.interactor.bias.data.fill_(0)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.beta_linear.weight)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.beta_linear.weight, mean=0.0, std=std)
        self.beta_linear.weight.data.mul_(0.01)
        self.beta_linear.bias.data.fill_(0.0)

        self.current_option = None

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        s, _, _, _ = self.level1_encoder(inputs)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        s2 = s
        s2, _, _, _ = self.level2_encoder(s2)
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat) * 0
        V2 = self.critic_linear2(s2_flat)
        beta = torch.sigmoid(self.beta_linear(s2_flat))

        a2_probs = F.softmax(a2_logits, dim=1)

        prev_option = self.current_option

        if prev_option is None or prev_option.size(0) != a2_probs.size(0):
            a2_sample = a2_probs.multinomial(1)
        else:
            beta_active = (beta.detach() * prev_option).sum(dim=1, keepdim=True)
            terminate = torch.bernoulli(beta_active)
            prev_idx = prev_option.argmax(dim=1, keepdim=True)
            new_idx = a2_probs.multinomial(1)
            a2_sample = torch.where(terminate.bool(), new_idx, prev_idx)

        a2_onehot = torch.zeros_like(a2_probs) * 0
        a2_onehot.scatter_(1, a2_sample, 1.0)
        self.current_option = a2_onehot.detach()

        if prev_option is not None and prev_option.size(0) == a2_probs.size(0):
            beta_active_grad = (beta * prev_option).sum(dim=1, keepdim=True)
        else:
            beta_active_grad = torch.zeros(
                beta.size(0), 1, device=beta.device, dtype=beta.dtype
            )

        a_21_logits = self.interactor(a2_onehot * 0) * 0

        s_flat = s.view(s.size(0), -1)
        s_flat = F.relu(s_flat)
        a1_logits = self.actor_linear(s_flat)
        V1 = self.critic_linear(s_flat)

        combined_logits = a1_logits + a_21_logits.detach()

        return (V1, combined_logits, hx, cx, None, None, V2, a2_logits,
                a1_logits, a_21_logits, a2_sample, beta_active_grad)


class Hierarchial_interactor_options_zeroing(Hierarchial_interactor_options):
    """Options variant with interactor (a21) zeroed; a1 and a2 play normally."""

    def forward(self, inputs, hx, cx, mem=None, bootstrap_only=False):
        (
            V1, _, hx, cx, _, _, V2, a2_logits,
            a1_logits, a_21_logits, a2_sample, beta_active_grad, V_intr,
            option_terminated,
        ) = super().forward(inputs, hx, cx, mem, bootstrap_only=bootstrap_only)

        a_21_logits = torch.zeros_like(a_21_logits)
        combined_logits = a1_logits + a_21_logits.detach()

        return (
            V1, combined_logits, hx, cx, None, None, V2, a2_logits,
            a1_logits, a_21_logits, a2_sample, beta_active_grad, V_intr,
            option_terminated,
        )


class Hierarchial_interactor_options_zeroing2(Hierarchial_interactor_options):
    """Options variant with level-1 actor (a1) zeroed; a21 and a2 play normally."""

    def forward(self, inputs, hx, cx, mem=None, bootstrap_only=False):
        (
            V1, _, hx, cx, _, _, V2, a2_logits,
            a1_logits, a_21_logits, a2_sample, beta_active_grad, V_intr,
            option_terminated,
        ) = super().forward(inputs, hx, cx, mem, bootstrap_only=bootstrap_only)

        a1_logits = torch.zeros_like(a1_logits)
        combined_logits = a1_logits + a_21_logits.detach()

        return (
            V1, combined_logits, hx, cx, None, None, V2, a2_logits,
            a1_logits, a_21_logits, a2_sample, beta_active_grad, V_intr,
            option_terminated,
        )


class Hierarchial_levels(nn.Module):
    """Two-level hierarchy built from peer Level modules.

    Level 2: s2 -> pi2 / V2 / beta2 (options; sticky via beta2).
             upper_options_dim=0 (top level, no higher option).
    Level 1: concat(s1, a2_onehot) -> pi1 / V1 / beta1 (env actions; sticky via beta1).
             upper_options_dim=num_options (conditioned on active a2).

    Termination subordination: if beta2 terminates, beta1 is forced to
    terminate (resample L1 action without sampling beta1). beta1 is sampled
    only while the level-2 option continues.

    Returns HierarchialLevelsOutput (namedtuple; see field docs on that type).
    """

    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_levels, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n
        self.num_outputs = num_outputs
        self.num_options = getattr(args, 'num_options', 8)

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        feat1 = 64 * 4 * 4
        feat2 = 32 * 4 * 4

        self.level2 = Level(
            encoder=EncoderRules234_2(),
            feat_dim=feat2,
            n_actions=self.num_options,
            upper_options_dim=0,
        )
        self.level1 = Level(
            encoder=EncoderRules234(
                num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm
            ),
            feat_dim=feat1,
            n_actions=num_outputs,
            upper_options_dim=self.num_options,
        )

        self.train()

    @property
    def current_option(self):
        return self.level2.current_action

    @current_option.setter
    def current_option(self, value):
        self.level2.current_action = value
        if value is None:
            self.level1.current_action = None

    def reset_persistent_actions(self):
        self.level1.reset_action()
        self.level2.reset_action()

    def forward(self, inputs, hx, cx, mem=None, bootstrap_only=False):
        s1 = self.level1.encode(inputs)

        if self.monitor_s and not bootstrap_only:
            self.s_values.append(s1.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        s2 = self.level2.encode(s1)
        out2 = self.level2.forward_heads(s2, bootstrap_only=bootstrap_only)
        # Subordination: beta2=1 => force beta1=1 (resample L1, do not sample beta1).
        # Only when beta2=0 (continue L2 option) do we sample beta1.
        force_l1_terminate = (not bootstrap_only) and bool(out2.terminated)
        # a2_onehot is non-differentiable upper-option context for level-1 heads
        out1 = self.level1.forward_heads(
            s1,
            upper_options_onehot=out2.action_onehot.detach(),
            bootstrap_only=bootstrap_only,
            force_terminate=force_l1_terminate,
        )

        return HierarchialLevelsOutput(
            V1=out1.V,
            a1_logits=out1.logits,
            hx=hx,
            cx=cx,
            mem=None,
            x_restored=None,
            V2=out2.V,
            a2_logits=out2.logits,
            a1=out1.action_idx,
            a2=out2.action_idx,
            beta1=out1.beta_grad,
            beta2=out2.beta_grad,
            terminated1=out1.terminated,
            terminated2=out2.terminated,
        )

