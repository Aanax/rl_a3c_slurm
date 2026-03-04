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
    """Encoder following Rules 2, 3, 4 - used by A3CRules2378"""
    def __init__(self):
        super(EncoderRules234_2, self).__init__()
        self.conv1 = nn.Conv2d(64+64, 32+32, 3, stride=1, padding=1)
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
        s2 = F.relu(s)
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


class Hierarchical_memory(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(Hierarchical_memory, self).__init__()
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
        self.critic_linear2 = nn.Linear(32*4*4, 1)
        self.actor_linear2 = nn.Linear(32*4*4, 16)

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
        s2 = F.relu(s)
        # Concat s2 with relu(running memory) along channel dimension
        mem_relu = F.relu(self.running_mem)
        s2_input = torch.cat([s2, mem_relu], dim=1) # 64 + 64 at input
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



