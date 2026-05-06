from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - kept for compatibility with EncoderRules234"""
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
    """Encoder following Rules 2, 3, 4 - used by A3CRules2378"""
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
            if conv is self.conv1:
                fan = fan_in
            else:
                fan = (fan_in + fan_out) / 2
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        # Rule 2: 3 conv-relu-maxpool layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        # Rule 3: conv4-relu-maxpool
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        # Rule 4: RMS-norm (optional) - controlled by use_rmsnorm flag
        if self.use_rmsnorm:
            x = x.view(x.size(0), -1)
            x = self.rmsnorm(x)
        return x, None, None, None

class A3CRules2378(nn.Module):
    """
    A3C model following Rules 2, 3, 7, 8 (no RMS normalization by default).

    Architecture pipeline (noRMS version):
    1) 3 conv-relu-maxpool layers
    2) conv4-relu-maxpool
    3) relu-fc
    4) linear value & actor heads
    """
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        num_outputs = action_space.n
        # Rule 7: relu-fc
        self.fc = nn.Linear(1024, self.hidden_size)
        # Rule 8: linear value&actor heads
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)

        # Custom initialization for fc
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = (fan_in + fan_out) / 2
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)
        self.fc.bias.data.fill_(0)

        # Heads initialization Rule 8: gaussian init with only fan_in
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        x, _, _, _ = self.encoder(inputs)
        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        # Rule 7: relu-fc
        x = F.relu(self.fc(s))

        # Rule 8: linear heads
        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, None

class A3CRules2378_nofc(nn.Module):
    """Same as A3CRules2378 but without fc after conv layers - heads connect directly to encoder output."""
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378_nofc, self).__init__()
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        num_outputs = action_space.n
        # No fc layer - connect directly to heads from encoder output (1024)
        # Rule 8: linear value&actor heads
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, num_outputs)

        # Heads initialization Rule 8: gaussian init with only fan_in
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        x, _, _, _ = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        # No fc layer - directly to heads
        # Rule 8: linear heads
        return self.critic_linear(s), self.actor_linear(s), hx, cx, None, None

class EncoderRules234_2(nn.Module):
    """Encoder following Rules 2, 3, 4 - used by A3CRules2378"""
    def __init__(self):
        super(EncoderRules234_2, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1]#, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv1:
                fan = fan_in
            else:
                fan = (fan_in + fan_out) / 2
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        # s1 dim 1024 at input? non flat!!! 64*4*4

        x = F.relu(self.conv1(x))

        print(f"Final output shape: {x.shape}")
        return x, None, None, None

class EncoderRules234_2_mem(nn.Module):
    """Encoder following Rules 2, 3, 4 - used by A3CRules2378 with more aggressive channel compression"""
    def __init__(self):
        super(EncoderRules234_2_mem, self).__init__()
        # More aggressive channel compression: 128 -> 16 channels
        self.conv1 = nn.Conv2d(64+64, 32+32, 3, stride=1, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv1:
                fan = fan_in
            else:
                fan = (fan_in + fan_out) / 2
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        # s1 dim 1024 at input? non flat!!! 64*4*4
        
        x = F.relu(self.conv1(x))

        print(f"Final output shape: {x.shape}")
        return x, None, None, None


class EncoderRules234_2_mem_A1(nn.Module):
    """Encoder following Rules 2, 3, 4 - used by A3CRules2378 with more aggressive channel compression"""
    def __init__(self, num_outputs=6):
        super(EncoderRules234_2_mem_A1, self).__init__()
        # More aggressive channel compression: (64+64+num_outputs) -> (32+32) channels
        self.num_outputs = num_outputs
        self.conv1 = nn.Conv2d(64+64+num_outputs, 32+32, 3, stride=1, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv1:
                fan = fan_in
            else:
                fan = (fan_in + fan_out) / 2
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        # s1 dim 1024 at input? non flat!!! 64*4*4

        x = F.relu(self.conv1(x))

        print(f"Final output shape: {x.shape}")
        return x, None, None, None

def action_vector_to_spatial(action_vector, spatial_h=4, spatial_w=4, apply_relu=True):
    """
    Convert action vector to spatial representation.

    Args:
        action_vector: Action vector of shape (batch_size, num_actions)
        spatial_h: Height of spatial representation
        spatial_w: Width of spatial representation
        apply_relu: Whether to apply ReLU activation

    Returns:
        Spatial representation of shape (batch_size, num_actions, spatial_h, spatial_w)
    """
    batch_size, num_actions = action_vector.size()
    action_spatial = []
    for i in range(num_actions):
        channel = action_vector[:, i:i+1]  # (batch, 1)
        channel = channel.repeat(1, spatial_h * spatial_w)  # (batch, spatial_h*spatial_w)
        channel = channel.view(channel.size(0), 1, spatial_h, spatial_w)  # (batch, 1, spatial_h, spatial_w)
        action_spatial.append(channel)
    action_spatial = torch.cat(action_spatial, dim=1)  # (batch, num_actions, spatial_h, spatial_w)
    if apply_relu:
        action_spatial = F.relu(action_spatial)
    return action_spatial

def normalize_action_memory(action_vector, spatial_h=4, spatial_w=4):
    """
    Normalize action memory vector using L2 normalization with scaling.

    Formula: (A/norm_L2(A)) * sqrt(dim(A)) where dim(A) is full feature map dimension.

    Args:
        action_vector: Action vector of shape (batch_size, num_actions)
        spatial_h: Height of spatial representation (nouse)
        spatial_w: Width of spatial representation

    Returns:
        Normalized action vector of shape (batch_size, num_actions)
    """
    # Calculate L2 norm along the action dimension (dim=1)
    l2_norm = torch.norm(action_vector, p=2, dim=1, keepdim=True)

    # Calc the full dim: num_actions * spatial_h * spatial_w
    full_dim = action_vector.view(action_vector.size(0), -1)
    full_dim = full_dim.size(1)
    scaling_factor = math.sqrt(full_dim)

    # Apply normalization: (A/norm_L2(A)) * sqrt(dim(A))
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    normalized = (action_vector / (l2_norm + epsilon)) * scaling_factor

    return normalized

class EncoderRules234_6more(nn.Module):
    """Encoder following Rules 2, 3, 4 - same as EncoderRules234 but taking 6 more channels at input"""
    def __init__(self, num_inputs, latent_dim_conv=64, use_rmsnorm=False):
        super(EncoderRules234_6more, self).__init__()
        self.use_rmsnorm = use_rmsnorm
        self.conv1 = nn.Conv2d(num_inputs + 6, 32, 5, stride=1, padding=2)
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
            if conv is self.conv1:
                fan = fan_in
            else:
                fan = (fan_in + fan_out) / 2
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        # Rule 2: 3 conv-relu-maxpool layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        # Rule 3: conv4-relu-maxpool
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        # Rule 4: RMS-norm (optional) - controlled by use_rmsnorm flag
        if self.use_rmsnorm:
            x = x.view(x.size(0), -1)
            x = self.rmsnorm(x)
        return x, None, None, None

class Hierarchial(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n

        # Level 1 encoder (same as A3CRules2378)
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        # Level 2 encoder (64*4*4 input)
        self.level2_encoder = EncoderRules234_2()

        # Level 2 heads (32*4*4 = 512 input)
        self.critic_linear2 = nn.Linear(32*4*4, 1)
        self.actor_linear2 = nn.Linear(32*4*4, 16)

        # Level 1 heads (64*4*4 = 1024 input, concat with a2)
        self.critic_linear = nn.Linear(64*4*4, 1)
        self.actor_linear = nn.Linear(64*4*4 + 16, num_outputs)

        # Initialize level 2 heads
        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        # Initialize level 1 heads
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        # Level 1 encoding
        s, _, _, _ = self.level1_encoder(inputs)  # s: 64*4*4

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        # Level 2 processing
        s2 = s #F.relu(s)
        s2, _, _, _ = self.level2_encoder(s2)  # s2: 32*4*4
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)

        # Sample from a2 probabilities to get one-hot binary vector
        a2_probs = F.softmax(a2_logits, dim=1)
        a2_sample = a2_probs.multinomial(1)  # Sample
        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)  # Create binary

        # Level 1 processing
        s_flat = s.view(s.size(0), -1)
        s_flat = F.relu(s_flat)
        actor_input = torch.cat([s_flat, a2_onehot], dim=1)
        a1 = self.actor_linear(actor_input)
        V1 = self.critic_linear(s_flat)

        return V1, a1, hx, cx, None, None, V2, a2_logits

class Hierarchial_memory(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_memory, self).__init__()
        self.hidden_size = args.hidden_size
        self.gamma1 = args.gamma
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n

        # Level 1 encoder (same as A3CRules2378)
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        # Level 2 encoder (64*4*4 input)
        self.level2_encoder = EncoderRules234_2_mem()

        # Level 2 heads (32*4*4 = 512 input)
        self.critic_linear2 = nn.Linear((32+32)*4*4, 1)
        self.actor_linear2 = nn.Linear((32+32)*4*4, 16)

        # Level 1 heads (64*4*4 = 1024 input + 1024 memory)
        self.critic_linear = nn.Linear(1024 + 1024, 1)
        self.actor_linear = nn.Linear(64*4*4 + 16, num_outputs)

        # Initialize level 2 heads
        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        # Initialize level 1 heads
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        # Internal memory for running differences on features (level1 only)
        self.running_mem = torch.zeros((1,64,4,4))
        self.prev_x_conv = torch.zeros((1,64,4,4))

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        # Level 1 encoding
        s, _, _, _ = self.level1_encoder(inputs)  # s: 64*4*4

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x_conv = s  # Keep as 4D tensor (64,4,4)

        # Compute running mem (level1 only)
        if self.running_mem is None:
            self.running_mem = torch.zeros_like(x_conv)
        if self.prev_x_conv is not None:
            diff = (x_conv - self.prev_x_conv).detach()
            self.running_mem = (diff + self.gamma1 * self.running_mem).detach()

        self.prev_x_conv = x_conv.detach()

        # Level 2
        # Level 2 processing with memory
        s2 = s#F.relu(s)
        # Concat s2 with relu(running memory) along channel dimension
        #F.relu(self.running_mem)
        s2_input = torch.cat([s2, self.running_mem], dim=1) # 64 + 64 at input
        s2, _, _, _ = self.level2_encoder(s2_input)  # s2: (32+32)*4*4 = 64*4*4
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)

        # Sample from a2 probabilities to get one-hot binary vector
        a2_probs = F.softmax(a2_logits, dim=1)
        a2_sample = a2_probs.multinomial(1)  # Sample
        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)  # Create binary

        # Level 1 processing
        # Flatten features for linear layers but keep memory as 4D
        s_flat = x_conv.view(x_conv.size(0), -1)
        critic_input = torch.cat([s_flat, self.running_mem.view(self.running_mem.size(0), -1)], dim=1)
        V1 = self.critic_linear(critic_input)
        actor_input = torch.cat([s_flat, a2_onehot], dim=1)
        a1 = self.actor_linear(actor_input)

        return V1, a1, hx, cx, None, None, V2, a2_logits

class Hierarchial_memory_memrelu(nn.Module):
    """
    Same as Hierarchial_memory but makes copies of memory before passing to critic and 2nd level,
    and passes these copies through ReLU activation.
    """
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_memory_memrelu, self).__init__()
        self.hidden_size = args.hidden_size
        self.gamma1 = args.gamma
        self.num_outputs = action_space.n
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n

        # Level 1 encoder (same as A3CRules2378)
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        # Level 2 encoder (64*4*4 input)
        self.level2_encoder = EncoderRules234_2_mem()

        # Level 2 heads ((32+32)*4*4 = 512 input)
        self.critic_linear2 = nn.Linear((32+32)*4*4, 1)
        self.actor_linear2 = nn.Linear((32+32)*4*4, 16)

        # Level 1 heads (64*4*4 = 1024 input + 1024 memory)
        self.critic_linear = nn.Linear(1024 + 1024, 1)
        self.actor_linear = nn.Linear(64*4*4 + 16, num_outputs)

        # Initialize level 2 heads
        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        # Initialize level 1 heads
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        # Internal memory for running differences on features (level1 only)
        self.running_mem = torch.zeros((1,64,4,4))
        self.prev_x_conv = torch.zeros((1,64,4,4))

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        # Level 1 encoding
        s, _, _, _ = self.level1_encoder(inputs)  # s: 64*4*4

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x_conv = s  # Keep as 4D tensor (64,4,4)

        # Compute running mem (level1 only)
        if self.running_mem is None:
            self.running_mem = torch.zeros_like(x_conv)
        if self.prev_x_conv is not None:
            diff = (x_conv - self.prev_x_conv).detach()
            self.running_mem = (diff + self.gamma1 * self.running_mem).detach()

        self.prev_x_conv = x_conv.detach()

        # Create copies of memory and pass through ReLU
        mem_for_level2 = F.relu(self.running_mem.clone())
        mem_for_critic = F.relu(self.running_mem.clone())

        # Level 2 processing with memory copy through ReLU
        s2 = s
        s2_input = torch.cat([s2, mem_for_level2], dim=1)  # 64 + 64 at input
        s2, _, _, _ = self.level2_encoder(s2_input)  # s2: (32+32)*4*4 = 64*4*4
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)

        # Sample from a2 probabilities to get one-hot binary vector
        a2_probs = F.softmax(a2_logits, dim=1)
        a2_sample = a2_probs.multinomial(1)  # Sample
        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)  # Create binary

        # Level 1 processing
        # Flatten features for linear layers
        s_flat = x_conv.view(x_conv.size(0), -1)
        # Use memory copy through ReLU for critic
        critic_input = torch.cat([s_flat, mem_for_critic.view(mem_for_critic.size(0), -1)], dim=1)
        V1 = self.critic_linear(critic_input)
        actor_input = torch.cat([s_flat, a2_onehot], dim=1)
        a1 = self.actor_linear(actor_input)

        return V1, a1, hx, cx, None, None, V2, a2_logits

class Hierarchial_memory_memrelu_no_a2(nn.Module):
    """
    Same as Hierarchial_memory but makes copies of memory before passing to critic and 2nd level,
    and passes these copies through ReLU activation.
    """
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_memory_memrelu_no_a2, self).__init__()
        self.hidden_size = args.hidden_size
        self.gamma1 = args.gamma
        self.num_outputs = action_space.n
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n

        # Level 1 encoder (same as A3CRules2378)
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        # Level 2 encoder (64*4*4 input)
        self.level2_encoder = EncoderRules234_2_mem()

        # Level 2 heads ((32+32)*4*4 = 512 input)
        self.critic_linear2 = nn.Linear((32+32)*4*4, 1)
        self.actor_linear2 = nn.Linear((32+32)*4*4, 16)

        # Level 1 heads (64*4*4 = 1024 input + 1024 memory)
        self.critic_linear = nn.Linear(1024 + 1024, 1)
        self.actor_linear = nn.Linear(64*4*4, num_outputs)

        # Initialize level 2 heads
        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        # Initialize level 1 heads
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        # Internal memory for running differences on features (level1 only)
        self.running_mem = torch.zeros((1,64,4,4))
        self.prev_x_conv = torch.zeros((1,64,4,4))

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        # Level 1 encoding
        s, _, _, _ = self.level1_encoder(inputs)  # s: 64*4*4

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x_conv = s  # Keep as 4D tensor (64,4,4)

        # Compute running mem (level1 only)
        if self.running_mem is None:
            self.running_mem = torch.zeros_like(x_conv)
        if self.prev_x_conv is not None:
            diff = (x_conv - self.prev_x_conv).detach()
            self.running_mem = (diff + self.gamma1 * self.running_mem).detach()

        self.prev_x_conv = x_conv.detach()

        # Create copies of memory and pass through ReLU
        mem_for_level2 = F.relu(self.running_mem.clone())
        mem_for_critic = F.relu(self.running_mem.clone())

        # Level 2 processing with memory copy through ReLU
        s2 = s
        s2_input = torch.cat([s2, mem_for_level2], dim=1)  # 64 + 64 at input
        s2, _, _, _ = self.level2_encoder(s2_input)  # s2: (32+32)*4*4 = 64*4*4
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)

        # Sample from a2 probabilities to get one-hot binary vector
        a2_probs = F.softmax(a2_logits, dim=1)
        a2_sample = a2_probs.multinomial(1)  # Sample
        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)  # Create binary

        # Level 1 processing
        # Flatten features for linear layers
        s_flat = x_conv.view(x_conv.size(0), -1)
        # Use memory copy through ReLU for critic
        critic_input = torch.cat([s_flat, mem_for_critic.view(mem_for_critic.size(0), -1)], dim=1)
        V1 = self.critic_linear(critic_input)
        actor_input = torch.cat([s_flat], dim=1)
        a1 = self.actor_linear(actor_input)

        return V1, a1, hx, cx, None, None, V2, a2_logits

class Hierarchial_memory_action_memrelu(nn.Module):
    """
    Same as Hierarchial_memory_memrelu but with action memory added.

    Action memory is computed as: a_t-1 + g*a_t-2 + g^2*a_t-3 + ...
    where g is the discount factor (gamma1).

    The previous action a_t-1 is passed as input to the forward method as a scalar index (0-5).
    Action memory is computed in the same way as state memory and is passed to:
    - Level 2 encoder (concatenated with state features)
    - Level 1 critic (concatenated with flattened state features)
    - Level 1 actor (passed through FC layer, then concatenated with a2_onehot)
    """
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_memory_action_memrelu, self).__init__()
        self.hidden_size = args.hidden_size
        self.gamma1 = args.gamma
        self.gamma2 = getattr(args, 'gamma2', 0.99)  # Default gamma2 value
        self.num_outputs = action_space.n

        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        # Level 1 encoder - use EncoderRules234_6more to handle 6 additional input channels
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234_6more(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        # Extract num_outputs before using it
        num_outputs = action_space.n

        # Level 2 encoder needs to handle additional A1 channels: state + state memory + action spatial + A1 spatial
        # Original: (64+64+num_outputs) -> (32+32+num_outputs)
        # New: (64+64+num_outputs+num_outputs) -> (32+32+num_outputs)
        self.level2_encoder = EncoderRules234_2_mem_A1(num_outputs)

        # Level 2 heads ((32+32)*4*4 = 512 input)
        self.critic_linear2 = nn.Linear((32+32)*4*4 + 16*4*4, 1)
        self.actor_linear2 = nn.Linear((32+32)*4*4, 16)

        # Level 1 heads (64*4*4 = 1024 input + 1024 memory)
        self.critic_linear = nn.Linear(1024 + 1024 + num_outputs*4*4, 1)
        self.actor_linear = nn.Linear(64*4*4 + 16, num_outputs)

        # Initialize level 2 encoder
        gain = nn.init.calculate_gain('relu')
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.level2_encoder.conv1.weight)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.level2_encoder.conv1.weight.uniform_(-bound, bound)
            self.level2_encoder.conv1.bias.data.fill_(0)

        # Initialize level 2 heads
        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        # Initialize level 1 heads
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        # Internal memory for running differences on state features (level1 only)
        self.running_mem = torch.zeros((1,64,4,4))
        self.prev_x_conv = torch.zeros((1,64,4,4))

        # Internal memory for action (stores vector of size num_outputs)
        self.A1 = torch.zeros((1, self.num_outputs))
        self.A2 = torch.zeros((1, 16))

        self.a2_prev = torch.zeros((1, 16))

        self.train()

    def forward(self, inputs, hx, cx, mem=None, action_prev=None):
        """
        Forward pass with action memory.
        """
        if action_prev is not None:
            a1_prev = action_prev  # Shape: (batch_size, num_outputs)
        else:
            batch_size = inputs.size(0)
            a1_prev = torch.zeros((batch_size, self.num_outputs), device=inputs.device)

        # Convert action_prev vector to spatial channels
        spatial_h, spatial_w = inputs.size(2), inputs.size(3)
        a1_prev_spatial = action_vector_to_spatial(a1_prev, spatial_h, spatial_w, apply_relu=False)

        # Concat a1_prev_spatial with inputs along channel dim
        encoder_input = torch.cat([inputs, a1_prev_spatial], dim=1)  # (batch, num_inputs+6, H, W)
        s, _, _, _ = self.level1_encoder(encoder_input)  # s: 64*4*4

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x_conv = s  # Keep as 4D tensor (64,4,4)

        # Compute running state mem (same as Hierarchial_memory_memrelu)
        if self.running_mem is None:
            self.running_mem = torch.zeros_like(x_conv)
        if self.prev_x_conv is not None:
            diff = (x_conv - self.prev_x_conv).detach()
            self.running_mem = (diff + self.gamma1 * self.running_mem).detach()

        self.prev_x_conv = x_conv.detach()

        # Copute action memory A1: a_t-1 + g*a_t-2 + g^2*a_t-3 + ...
        # This is the cumulative action memory vector
        if self.A1 is None:
            self.A1 = torch.zeros((x_conv.size(0), self.num_outputs), device=x_conv.device)
        # A1 = a_t-1 (a_prev) + gamma1 * A1
        self.A1 = (a1_prev + self.gamma1 * self.A1).detach()

        # Compute action memory A2: similar to A1 but with gamma2
        if self.A2 is None:
            self.A2 = torch.zeros((x_conv.size(0), self.num_outputs), device=x_conv.device)
        # A2 = a_t-1 (a_prev) + gamma2 * previous_running_memory_action2
        self.A2 = (self.a2_prev + self.gamma2 * self.A2).detach() #dim 16

        # Normalize A1 and A2 using L2 normalization with scaling
        # Formula: (A/norm_L2(A)) * sqrt(dim(A)) where dim(A) is full feature map dimension
        # A1_normalized = normalize_action_memory(self.A1.clone())
        # A2_normalized = normalize_action_memory(self.A2.clone())

        # Create copies of state memory and action memory through ReLU
        mem_for_level2 = F.relu(self.running_mem.clone())
        mem_for_critic = F.relu(self.running_mem.clone())

        # Level 2
        s2 = s
        A1_spatial_normalized = action_vector_to_spatial(self.A1.clone(), spatial_h=s2.size(2), spatial_w=s2.size(3), apply_relu=False)
        A1_spatial_normalized = normalize_action_memory(A1_spatial_normalized)
        A2_spatial_normalized = action_vector_to_spatial(self.A2.clone(), spatial_h=s2.size(2), spatial_w=s2.size(3), apply_relu=False)
        A2_spatial_normalized = normalize_action_memory(A2_spatial_normalized)

        # Concatenate state, state memory, action spatial, and A1 spatial along channel dimension
        s2_input = torch.cat([s2, mem_for_level2, A1_spatial_normalized], dim=1)  # 64 + 64 + num_outputs + num_outputs channels
        s2, _, _, _ = self.level2_encoder(s2_input)  # s2: (32+32)*4*4 = 64*4*4 = 1024
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)

        V2 = self.critic_linear2(torch.cat([s2_flat, A2_spatial_normalized.view(A2_spatial_normalized.size(0),-1)], dim=1))

        # Sample from a2 probabilities to get one-hot binary vector
        a2_probs = F.softmax(a2_logits, dim=1)
        a2_sample = a2_probs.multinomial(1)  # Sample
        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)  # Create binary

        self.a2_prev = a2_onehot.detach()

        s_flat = x_conv.view(x_conv.size(0), -1)  # Shape: (batch, 1024)
        critic_input = torch.cat([
            s_flat,
            mem_for_critic.view(mem_for_critic.size(0), -1),
            A1_spatial_normalized.view(A1_spatial_normalized.size(0), -1),
        ], dim=1)  # (batch, 1024 + 1024 + num_outputs)

        V1 = self.critic_linear(critic_input)
        # Actor uses state + a2_onehot + action memory vector (A1)
        actor_input = torch.cat([s_flat, a2_onehot], dim=1)  # (batch, 1024 + 16
        a1 = self.actor_linear(actor_input)

        return V1, a1, hx, cx, None, None, V2, a2_logits

class Hierarchial_a2a1_connect(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchial_a2a1_connect, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        num_outputs = action_space.n

        # Level 1 encoder (same as A3CRules2378)
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.level1_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        # Level 2 encoder (64*4*4 input)
        self.level2_encoder = EncoderRules234_2()

        # Level 2 heads (32*4*4 = 512 input)
        self.critic_linear2 = nn.Linear(32*4*4, 1)
        self.actor_linear2 = nn.Linear(32*4*4, num_outputs)

        # Level 1 heads (64*4*4 = 1024 input, concat with a2)
        self.critic_linear = nn.Linear(64*4*4, 1)
        self.actor_linear = nn.Linear(64*4*4 + num_outputs, num_outputs)

        # Initialize level 2 heads
        for linear in [self.critic_linear2, self.actor_linear2]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        # Initialize level 1 heads
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        # Level 1 encoding
        s, _, _, _ = self.level1_encoder(inputs)  # s: 64*4*4

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        # Level 2 processing
        s2 = s #F.relu(s)
        s2, _, _, _ = self.level2_encoder(s2)  # s2: 32*4*4
        s2_flat = s2.view(s2.size(0), -1)
        a2_logits = self.actor_linear2(s2_flat)
        V2 = self.critic_linear2(s2_flat)

        # Sample from a2 probabilities to get one-hot binary vector
        a2_probs = F.softmax(a2_logits, dim=1)
        a2_sample = a2_probs.multinomial(1)  # Sample
        a2_onehot = torch.zeros_like(a2_probs)
        a2_onehot.scatter_(1, a2_sample, 1.0)  # Create binary

        # Level 1 processing
        s_flat = s.view(s.size(0), -1)
        s_flat = F.relu(s_flat)
        actor_input = torch.cat([s_flat, a2_onehot], dim=1)
        a1 = self.actor_linear(actor_input)
        V1 = self.critic_linear(s_flat)

        return V1, a1, hx, cx, None, None, V2, a2_logits