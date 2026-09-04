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


class DecoderRules234(nn.Module):
    def __init__(self, num_inputs, use_rmsnorm=False, latent_dim=64):
        super(DecoderRules234, self).__init__()
        self.use_rmsnorm = use_rmsnorm
        # Mirror of EncoderRules234 convolutions in reverse order.
        self.deconv4 = nn.ConvTranspose2d(latent_dim, 64, 3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, 5, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, num_inputs, 5, stride=1, padding=2)

    def forward(self, x):
        if self.use_rmsnorm and x.dim() == 2:
            spatial = int(math.sqrt(x.size(1) // 64))
            x = x.view(x.size(0), 64, spatial, spatial)
        # Undo max-pool stages with target spatial sizes for 80x80 inputs.
        x = F.interpolate(x, size=(9, 9), mode="nearest")
        x = F.relu(self.deconv4(x))
        x = F.interpolate(x, size=(18, 18), mode="nearest")
        x = F.relu(self.deconv3(x))
        x = F.interpolate(x, size=(38, 38), mode="nearest")
        x = F.relu(self.deconv2(x))
        x = F.interpolate(x, size=(80, 80), mode="nearest")
        x = self.deconv1(x)
        return x


class DecoderRules234FC(DecoderRules234):
    """Decoder for FC features: hidden -> 1024 -> mirrored DecoderRules234 topology."""
    def __init__(self, hidden_size, num_inputs):
        super(DecoderRules234FC, self).__init__(num_inputs, use_rmsnorm=False)
        self.fc = nn.Linear(hidden_size, 1024)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, 4, 4)
        return super(DecoderRules234FC, self).forward(x)


class A3CRules2378Oracle(A3CRules2378):
    """A3CRules2378Mem with decoder and addtional loss.

    1. Делаем декодер и добавляем лосс как в oracle (без fc)
    2. Восстанавлиаем из fc
    3. Передаем выход декодера в критика

    делаем сначала без памяти
    батч сайз 32
    4. Добавляем память

    The decoder reconstructs the input from the encoder feature map or RMSNorm-flattened
    representation, and the forward returns x_restored for the restoration loss in train.py.
    """
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378Oracle, self).__init__(num_inputs, action_space, args)
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []
        
        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)
        self.decoder = DecoderRules234(num_inputs, use_rmsnorm=use_rmsnorm)

        num_outputs = action_space.n
        # Oracle path: actor/critic/decoder consume encoder features directly.
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

        x_restored = self.decoder(x)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        return self.critic_linear(s), self.actor_linear(s), hx, cx, x_restored


class A3CRules2378OracleFC(A3CRules2378Oracle):
    """A3CRules2378Oracle variant that decodes from FC activations instead of encoder output."""
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378OracleFC, self).__init__(num_inputs, action_space, args)
        num_outputs = action_space.n
        self.fc = nn.Linear(1024, self.hidden_size)
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.decoder = DecoderRules234FC(self.hidden_size, num_inputs)

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = (fan_in + fan_out) / 2
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)
        self.fc.bias.data.fill_(0)

        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)

        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

    def forward(self, inputs, hx, cx, mem=None):
        x, _, _, _ = self.encoder(inputs)
        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))
        x_restored = self.decoder(x)

        return self.critic_linear(x), self.actor_linear(x), hx, cx, x_restored


class A3CRules2378OracleIntrinsicCritic(A3CRules2378Oracle):
    """A3CRules2378Oracle with an additional intrinsic critic head."""
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378OracleIntrinsicCritic, self).__init__(num_inputs, action_space, args)
        self.critic_linear_intrinsic = nn.Linear(1024, 1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.critic_linear_intrinsic.weight)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.critic_linear_intrinsic.weight, mean=0.0, std=std)
        self.critic_linear_intrinsic.bias.data.fill_(0)
        self.critic_linear_intrinsic.weight.data.mul_(1.0)

    def forward(self, inputs, hx, cx, mem=None):
        x, _, _, _ = self.encoder(inputs)
        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        x_restored = self.decoder(x)
        hx = torch.Tensor([0])
        cx = torch.Tensor([0])
        value_intrinsic = self.critic_linear_intrinsic(s)
        return self.critic_linear(s), self.actor_linear(s), hx, cx, x_restored, value_intrinsic


class A3CRules2378OracleFCIntrinsicCritic(A3CRules2378OracleFC):
    """A3CRules2378OracleFC with an additional intrinsic critic head."""
    def __init__(self, num_inputs, action_space, args):
        super(A3CRules2378OracleFCIntrinsicCritic, self).__init__(num_inputs, action_space, args)
        self.critic_linear_intrinsic = nn.Linear(self.hidden_size, 1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.critic_linear_intrinsic.weight)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.critic_linear_intrinsic.weight, mean=0.0, std=std)
        self.critic_linear_intrinsic.bias.data.fill_(0)
        self.critic_linear_intrinsic.weight.data.mul_(1.0)

    def forward(self, inputs, hx, cx, mem=None):
        x, _, _, _ = self.encoder(inputs)
        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])
        x = F.relu(self.fc(s))
        x_restored = self.decoder(x)
        value_intrinsic = self.critic_linear_intrinsic(x)
        return self.critic_linear(x), self.actor_linear(x), hx, cx, x_restored, value_intrinsic


class A3CRules2378OracleNoSplitEncoders(nn.Module):
    """Oracle topology WITHOUT per-branch encoders (public base, usable as a `model_type`).

    Pipeline: shared encoder -> `_branch_inputs` routing -> actor/critic heads read the branch
    features directly (no per-branch conv); the decoder reconstructs from the actor/oracle
    branch features. Mix in:
      * `_SplitEncodersMixin`     -> adds the per-branch conv encoders (the classic
                                     `A3CRules2378OracleSplitEncoders` topology),
      * `_SharedFeatureDiffMixin` -> diff/concat input routing via actor_input_mode/critic_input_mode,
      * `_IntrinsicCriticMixin`   -> an extra intrinsic critic head.

    Head/decoder widths follow each branch's input channels (`_*_branch_channels`): 'shared'/'diff'
    -> 64 ch / 1024 flat, 'concat' -> 128 ch / 2048 flat. Used bare, both heads read the current
    shared features (64 ch)."""
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []

        use_rmsnorm = getattr(args, 'use_rmsnorm', False)
        self.shared_encoder = EncoderRules234(num_inputs, latent_dim_conv=64, use_rmsnorm=use_rmsnorm)

        actor_ch = self._actor_branch_channels()
        critic_ch = self._critic_branch_channels()

        # Decoder restores from the actor/oracle branch features (actor_ch channels, 4x4).
        self.decoder = DecoderRules234(num_inputs, use_rmsnorm=False, latent_dim=actor_ch)

        num_outputs = action_space.n
        self.actor_linear = nn.Linear(actor_ch * 16, num_outputs)
        self.critic_linear = nn.Linear(critic_ch * 16, 1)
        self._init_heads()
        self.train()

    # --- dim hooks: _SplitEncodersMixin overrides these to 64 (its conv output width) ---
    def _actor_branch_channels(self):
        return self._mode_channels(getattr(self, 'actor_input_mode', 'shared'))

    def _critic_branch_channels(self):
        return self._mode_channels(getattr(self, 'critic_input_mode', 'shared'))

    @staticmethod
    def _mode_channels(mode):
        return 128 if mode in ('concat', 'concat_memdiff', 'concat_memdiff_raw') else 64

    def _init_heads(self):
        for linear in [self.critic_linear, self.actor_linear]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            std = 1.0 / math.sqrt(fan_in)
            nn.init.normal_(linear.weight, mean=0.0, std=std)
            linear.bias.data.fill_(0)
        self.actor_linear.weight.data.mul_(0.01)
        self.critic_linear.weight.data.mul_(1.0)

    @staticmethod
    def _init_branch_conv(conv):
        if conv.bias is not None:
            conv.bias.data.fill_(0)
        gain = nn.init.calculate_gain('relu')
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
        fan = (fan_in + fan_out) / 2
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        nn.init.uniform_(conv.weight, -bound, bound)

    # --- routing / encoding hooks (overridable by mixins) ---
    def _branch_inputs(self, shared):
        """Inputs for the (actor/oracle, critic) branches. Base feeds current shared to both;
        _SharedFeatureDiffMixin overrides with diff/concat routing."""
        return shared, shared

    def _encode_actor(self, x):
        """Base (no split): identity. _SplitEncodersMixin overrides with relu(conv)."""
        return x

    def _encode_critic(self, x):
        return x

    def _extra_outputs(self, actor_flat):
        """Base: no extra head. _IntrinsicCriticMixin overrides to append value_intrinsic."""
        return ()

    def _forward_core(self, inputs):
        shared, _, _, _ = self.shared_encoder(inputs)
        shared = shared.view(shared.size(0), 64, 4, 4)

        actor_in, critic_in = self._branch_inputs(shared)
        actor_feat = self._encode_actor(actor_in)
        critic_feat = self._encode_critic(critic_in)

        actor_flat = actor_feat.reshape(actor_feat.size(0), -1)
        critic_flat = critic_feat.reshape(critic_feat.size(0), -1)

        if self.monitor_s:
            self.s_values.append(shared.detach().cpu())

        x_restored = self.decoder(actor_feat)
        return actor_flat, critic_flat, x_restored

    def forward(self, inputs, hx, cx, mem=None):
        actor_flat, critic_flat, x_restored = self._forward_core(inputs)
        hx = torch.Tensor([0])
        cx = torch.Tensor([0])
        return (
            self.critic_linear(critic_flat),
            self.actor_linear(actor_flat),
            hx,
            cx,
            x_restored,
            *self._extra_outputs(actor_flat),
        )


class _SplitEncodersMixin:
    """Mixin: adds the per-branch conv encoders (actor/oracle + critic), the classic
    "split encoders" topology. Each is a Conv2d(branch_in -> 64), so the heads and decoder
    always see 64 ch / 1024 flat regardless of the input mode. Mix in to the LEFT of the
    NoSplit base (and to the RIGHT of _SharedFeatureDiffMixin, so the modes are set first)."""
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        actor_in = self._mode_channels(getattr(self, 'actor_input_mode', 'shared'))
        critic_in = self._mode_channels(getattr(self, 'critic_input_mode', 'shared'))
        self.actor_oracle_encoder = nn.Conv2d(actor_in, 64, kernel_size=3, stride=1, padding=1)
        self.critic_encoder = nn.Conv2d(critic_in, 64, kernel_size=3, stride=1, padding=1)
        self._init_branch_conv(self.actor_oracle_encoder)
        self._init_branch_conv(self.critic_encoder)

    # branch convs output 64 ch -> heads/decoder are sized at 64 regardless of input mode
    def _actor_branch_channels(self):
        return 64

    def _critic_branch_channels(self):
        return 64

    def _encode_actor(self, x):
        return F.relu(self.actor_oracle_encoder(x))

    def _encode_critic(self, x):
        return F.relu(self.critic_encoder(x))


class _IntrinsicCriticMixin:
    """Mixin: adds an intrinsic critic head (read from the actor/oracle branch) via a 6th
    forward output. `isinstance(model, _IntrinsicCriticMixin)` is the canonical "has an
    intrinsic critic" test (used in train.py / player_util.py). The head is sized to the actor
    branch flat width, so it works for both split (1024) and no-split concat (2048)."""
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.critic_linear_intrinsic = nn.Linear(self._actor_branch_channels() * 16, 1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.critic_linear_intrinsic.weight)
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.critic_linear_intrinsic.weight, mean=0.0, std=std)
        self.critic_linear_intrinsic.bias.data.fill_(0)
        self.critic_linear_intrinsic.weight.data.mul_(1.0)

    def _extra_outputs(self, actor_flat):
        return (self.critic_linear_intrinsic(actor_flat),)


class _SharedFeatureDiffMixin:
    """Mixin: routes per-branch inputs via the temporal difference of shared-encoder features.
    Config-driven by `actor_input_mode` / `critic_input_mode` (from args), each one of:
        'shared' -> current shared features          (64 ch)
        'diff'   -> shared_cur - prev_shared         (64 ch)
        'concat' -> concat([shared, diff])           (128 ch, appearance + motion)
        'concat_memdiff' -> concat([shared, shared - normalized memory]) (128 ch)
        'concat_memdiff_raw' -> concat([shared, shared - (1-gamma)*memory]) (128 ch)
    Defaults: actor 'concat', critic 'shared'. Mix in to the LEFT of the base.

    Modes are set BEFORE super().__init__() so the base/split can size heads, decoder, and
    branch convs by mode. `prev_shared` is per-worker model state (each train worker holds a
    local model copy), reset to None at episode boundaries (see player_util) so the first frame
    of an episode yields a zero diff. It keeps the graph (no per-step detach), so the diff also
    backprops through the previous step (within-rollout BPTT); train.py detaches it at the batch
    boundary so the next batch's first diff doesn't backprop into the freed graph. Declared as a
    class attribute so hasattr(model, 'prev_shared') is True for the reset."""
    prev_shared = None
    memdiff_sum = None
    memdiff_count = 0
    _VALID_MODES = ('shared', 'diff', 'concat', 'concat_memdiff', 'concat_memdiff_raw')

    def __init__(self, num_inputs, action_space, args):
        self.actor_input_mode = getattr(args, 'actor_input_mode', 'concat')
        self.critic_input_mode = getattr(args, 'critic_input_mode', 'shared')
        self.gamma_memory = getattr(args, 'gamma_memory', 0.99)
        assert self.actor_input_mode in self._VALID_MODES, f"bad actor_input_mode: {self.actor_input_mode}"
        assert self.critic_input_mode in self._VALID_MODES, f"bad critic_input_mode: {self.critic_input_mode}"
        super().__init__(num_inputs, action_space, args)

    def _memory_diffs(self, shared):
        if self.memdiff_sum is None or self.memdiff_count == 0:
            normalized = raw = torch.zeros_like(shared)
        else:
            raw = shared - self.memdiff_sum * (1.0 - self.gamma_memory)
            if self.gamma_memory == 1.0:
                normalized = shared - (self.memdiff_sum / self.memdiff_count)
            else:
                weight = (1.0 - self.gamma_memory) / (1.0 - self.gamma_memory ** self.memdiff_count)
                normalized = shared - self.memdiff_sum * weight

        self.memdiff_sum = shared if self.memdiff_sum is None else shared + self.gamma_memory * self.memdiff_sum
        self.memdiff_count += 1
        return normalized, raw

    def _branch_inputs(self, shared):
        prev = self.prev_shared
        diff = torch.zeros_like(shared) if prev is None else shared - prev
        self.prev_shared = shared
        memory_modes = (self.actor_input_mode, self.critic_input_mode)
        normalized_memdiff = raw_memdiff = None
        if 'concat_memdiff' in memory_modes or 'concat_memdiff_raw' in memory_modes:
            normalized_memdiff, raw_memdiff = self._memory_diffs(shared)
        pick = {
            'shared': shared,
            'diff': diff,
            'concat': torch.cat([shared, diff], dim=1),
            'concat_memdiff': (
                torch.cat([shared, normalized_memdiff], dim=1)
                if normalized_memdiff is not None else None
            ),
            'concat_memdiff_raw': (
                torch.cat([shared, raw_memdiff], dim=1)
                if raw_memdiff is not None else None
            ),
        }
        return pick[self.actor_input_mode], pick[self.critic_input_mode]


# --- Split-encoder variants (classic topology: per-branch conv encoders). Names/configs unchanged. ---
class A3CRules2378OracleSplitEncoders(_SplitEncodersMixin, A3CRules2378OracleNoSplitEncoders):
    """Oracle split topology: NoSplit base + per-branch conv encoders (actor/oracle, critic)."""
    pass


class A3CRules2378OracleSplitEncodersIntrinsicCritic(_IntrinsicCriticMixin, A3CRules2378OracleSplitEncoders):
    """Split-encoder oracle model with an additional intrinsic critic head."""
    pass


class A3CRules2378OracleSplitEncodersSharedDiff(_SharedFeatureDiffMixin, A3CRules2378OracleSplitEncoders):
    """Split-encoder oracle model with diff/concat input routing (actor_input_mode /
    critic_input_mode); branch convs reduce each input back to 64 ch."""
    pass


class A3CRules2378OracleSplitEncodersSharedDiffIntrinsicCritic(_SharedFeatureDiffMixin, _IntrinsicCriticMixin, A3CRules2378OracleSplitEncoders):
    """SharedDiff split-encoder variant with an additional intrinsic critic head."""
    pass


# --- No-split variants (heads read shared/diff/concat directly; no per-branch conv). ---
class A3CRules2378OracleNoSplitSharedDiff(_SharedFeatureDiffMixin, A3CRules2378OracleNoSplitEncoders):
    """No-split oracle model with diff/concat input routing. With actor_input_mode=concat /
    critic_input_mode=concat this is the no-split + concat topology (heads read the 128-ch
    shared+diff concat directly, 2048 flat; decoder restores from the 128-ch concat)."""
    pass


class A3CRules2378OracleNoSplitSharedDiffIntrinsicCritic(_SharedFeatureDiffMixin, _IntrinsicCriticMixin, A3CRules2378OracleNoSplitEncoders):
    """No-split SharedDiff variant with an additional intrinsic critic head."""
    pass
