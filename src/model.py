"""
The neural network of the agent.

The agent has two levels. Both levels look at the same game screen.

  Level 2 (the "manager") picks an OPTION every step: a number from 0 to num_options-1.
  Level 1 (the "worker") picks the GAME ACTION every step, knowing the current option.

Each level has four heads:
  actor             - what to do (game action for level 1, option for level 2)
  external critic   - how much game reward is expected
  intrinsic critic  - how much oracle reward is expected
  oracle            - a guess of how the level's features will change next

Level 1 also has a MEMORY: besides the current features s1 it sees how s1 changed
since the previous step (s1 - previous s1).

Shapes (for one 80x80 frame):
  s1      64 x 4 x 4   features of the screen
  memory  64 x 4 x 4   s1 - previous s1
  s2      32 x 4 x 4   level-2 features, computed from s1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_conv(conv, use_fan_in_only):
    """Uniform init scaled for ReLU. Biases start at zero."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
    if use_fan_in_only:
        fan = fan_in
    else:
        fan = (fan_in + fan_out) / 2
    bound = math.sqrt(3.0) * nn.init.calculate_gain('relu') / math.sqrt(fan)
    nn.init.uniform_(conv.weight, -bound, bound)
    conv.bias.data.fill_(0)


def init_linear(linear, scale):
    """Gaussian init with std 1/sqrt(fan_in), then multiplied by `scale`. Biases start at zero."""
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
    nn.init.normal_(linear.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
    linear.weight.data.mul_(scale)
    linear.bias.data.fill_(0)


class EncoderRules234(nn.Module):
    """Screen (1 x 80 x 80) -> s1 (64 x 4 x 4). Four conv-maxpool-relu layers."""

    def __init__(self, num_inputs):
        super(EncoderRules234, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        init_conv(self.conv1, use_fan_in_only=True)
        init_conv(self.conv2, use_fan_in_only=False)
        init_conv(self.conv3, use_fan_in_only=False)
        init_conv(self.conv4, use_fan_in_only=False)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        return x


class EncoderRules234_2(nn.Module):
    """s1 (64 x 4 x 4) -> s2 (32 x 4 x 4). One conv-relu layer."""

    def __init__(self):
        super(EncoderRules234_2, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        init_conv(self.conv1, use_fan_in_only=True)

    def forward(self, x):
        return F.relu(self.conv1(x))


class A3CRules2378OracleTwoLevelMemory(nn.Module):
    """Two-level oracle agent with a one-step memory on level 1.

    Args:
        num_inputs: Number of screen channels (1).
        action_space: Game action space. Uses ``action_space.n``.
        args: Run config. Reads ``num_options``.
    """

    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378OracleTwoLevelMemory, self).__init__()
        self.num_options = args.num_options
        num_actions = action_space.n

        s1_channels = 64
        level1_channels = s1_channels * 2          # s1 and memory side by side
        level1_size = level1_channels * 4 * 4      # 2048 numbers
        s2_channels = 32
        s2_size = s2_channels * 4 * 4              # 512 numbers

        # Encoders
        self.shared_encoder = EncoderRules234(num_inputs)
        self.level2_encoder = EncoderRules234_2()

        # Level 1 heads. Each one also reads the option as a one-hot vector.
        self.actor_linear = nn.Linear(level1_size + self.num_options, num_actions)
        self.critic_linear = nn.Linear(level1_size + self.num_options, 1)
        self.critic_linear_intrinsic = nn.Linear(level1_size + self.num_options, 1)
        self.oracle1_head = nn.Conv2d(level1_channels + self.num_options, s1_channels, kernel_size=1)

        # Level 2 heads. They do not see the option: they choose it.
        self.actor_linear2 = nn.Linear(s2_size, self.num_options)
        self.critic_linear2 = nn.Linear(s2_size, 1)
        self.critic_linear_intrinsic2 = nn.Linear(s2_size, 1)
        self.oracle2_head = nn.Conv2d(s2_channels, s2_channels, kernel_size=1)

        init_linear(self.actor_linear, 0.01)
        init_linear(self.critic_linear, 1.0)
        init_linear(self.critic_linear_intrinsic, 1.0)
        init_conv(self.oracle1_head, use_fan_in_only=False)
        init_linear(self.actor_linear2, 0.01)
        init_linear(self.critic_linear2, 1.0)
        init_linear(self.critic_linear_intrinsic2, 1.0)
        init_conv(self.oracle2_head, use_fan_in_only=False)

        self.train()

    def forward(self, state, previous_s1, option=None):
        """Run both levels on one screen.

        Args:
            state: Screen, shape ``[1, 1, 80, 80]``.
            previous_s1: ``s1`` of the previous step, or None at the start of an episode.
            option: Option to use. When None the model chooses it: sampled while
                training, the most likely one in ``eval()`` mode.

        Returns:
            Dict with every output, see the end of this function.
        """
        s1 = self.shared_encoder(state)

        # Memory: how s1 changed since the previous step. Zero on the first step.
        # previous_s1 is not detached, so gradients also flow into the previous step.
        if previous_s1 is None:
            s1_change = torch.zeros_like(s1)
        else:
            s1_change = s1 - previous_s1
        level1_input = torch.cat([s1, s1_change], dim=1)
        level1_flat = level1_input.reshape(1, -1)

        # Level 2: features, heads, and oracle 2
        s2 = self.level2_encoder(s1)
        s2_flat = s2.reshape(1, -1)
        option_logits = self.actor_linear2(s2_flat)
        value2 = self.critic_linear2(s2_flat)
        value_intrinsic2 = self.critic_linear_intrinsic2(s2_flat)
        oracle2_prediction = self.oracle2_head(s2)

        # Choose the option, then write it as a one-hot vector: [0, 1, 0, 0, ...]
        if option is None:
            option_probs = F.softmax(option_logits, dim=1)
            if self.training:
                option = option_probs.multinomial(1)
            else:
                option = option_probs.argmax(dim=1, keepdim=True)
        option = torch.as_tensor(option).long().reshape(1, 1)
        option_onehot = torch.zeros(1, self.num_options)
        option_onehot[0, option.item()] = 1.0

        # Level 1: heads see the features, the memory, and the option
        level1_with_option = torch.cat([level1_flat, option_onehot], dim=1)
        action_logits = self.actor_linear(level1_with_option)
        value = self.critic_linear(level1_with_option)
        value_intrinsic = self.critic_linear_intrinsic(level1_with_option)

        # Oracle 1 gets the option as extra constant channels on the 4x4 map
        option_map = option_onehot[:, :, None, None].expand(-1, -1, 4, 4)
        oracle1_prediction = self.oracle1_head(torch.cat([level1_input, option_map], dim=1))

        return {
            # level 1
            'action_logits': action_logits,
            'value': value,
            'value_intrinsic': value_intrinsic,
            'oracle1_prediction': oracle1_prediction,   # guess of the change of s1
            # level 2
            'option_logits': option_logits,
            'option': option,
            'value2': value2,
            'value_intrinsic2': value_intrinsic2,
            'oracle2_prediction': oracle2_prediction,   # guess of the change of s2
            # features
            's1': s1,             # not detached: the next step's memory uses it
            's2': s2.detach(),
        }
