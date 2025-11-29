from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import norm_col_init, weights_init
from decoders import Decoder_AE_nobn


class Encoder(nn.Module):
    def __init__(self, num_inputs):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2.bias.data.fill_(0)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv3.bias.data.fill_(0)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)


class EncoderWithRelu(Encoder):
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))
        return x


class EncoderNoReluLast(Encoder):
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.max_pool2d(self.conv4(x), 2, 2)
        return x


class EncoderVAE(Encoder):
    def __init__(self, num_inputs, latent_dim=1024):
        super(EncoderVAE, self).__init__(num_inputs)
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.max_pool2d(self.conv4(x), 2, 2)

        x2 = x.view(x.size(0), -1)
        z_mean = self.fc_mu(x2)
        z_log_var = self.fc_var(x2)
        z = z_mean + torch.exp(z_log_var / 2) * self.N.sample(z_mean.shape)
        kl = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).mean()

        return z, z_mean, z_log_var, kl


class EncoderConvKL(Encoder):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super(EncoderConvKL, self).__init__(num_inputs)
        self.latent_dim = latent_dim_conv
        # Double the last conv to output 2*latent_dim channels for mean and log_var
        self.conv4 = nn.Conv2d(64, 2 * latent_dim_conv, 3, stride=1, padding=1)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.max_pool2d(self.conv4(x), 2, 2)  # x shape: (batch, 2*latent_dim, H, W)

        # Split into z_mean and z_log_var
        z_mean, z_log_var = torch.chunk(x, 2, dim=1)

        # Sample z
        noise = self.N.sample(z_mean.shape)
        z = z_mean + torch.exp(z_log_var / 2) * noise

        # KL divergence
        kl = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).mean()

        return z, z_mean, z_log_var, kl


class EncoderConvKLWithFanInOut(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()
        self.latent_dim = latent_dim_conv
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 2 * latent_dim_conv, 3, stride=1, padding=1)
        self.N = torch.distributions.Normal(0, 1)
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.max_pool2d(self.conv4(x), 2, 2)  # x shape: (batch, 2*latent_dim, H, W)

        # Split into z_mean and z_log_var
        z_mean, z_log_var = torch.chunk(x, 2, dim=1)

        # Sample z
        noise = self.N.sample(z_mean.shape)
        z = z_mean + torch.exp(z_log_var / 2) * noise

        # KL divergence
        kl = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).mean()

        return z, z_mean, z_log_var, kl


class EncoderConvKLWithFanOut1ForLastConv(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()
        self.latent_dim = latent_dim_conv
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 2 * latent_dim_conv, 3, stride=1, padding=1)
        self.N = torch.distributions.Normal(0, 1)
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv4:
                fan = fan_in + 1  # fan_out = 1 for conv4
            else:
                fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.max_pool2d(self.conv4(x), 2, 2)  # x shape: (batch, 2*latent_dim, H, W)

        # Split into z_mean and z_log_var
        z_mean, z_log_var = torch.chunk(x, 2, dim=1)

        # Sample z
        noise = self.N.sample(z_mean.shape)
        z = z_mean + torch.exp(z_log_var / 2) * noise

        # KL divergence
        kl = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).mean()

        return z, z_mean, z_log_var, kl


class EncoderConvWithFanOut1ForLastConvNoKL(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()
        self.latent_dim = latent_dim_conv
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, latent_dim_conv, 3, stride=1, padding=1)  # latent_dim_conv channels for no KL
        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv4:
                fan = fan_in + 1  # fan_out = 1 for conv4
            elif conv is self.conv1:
                fan = fan_in
            else:
                fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.max_pool2d(self.conv4(x), 2, 2)  # x shape: (batch, latent_dim, H, W)

        # No KL, just flatten
        x = x.view(x.size(0), -1)

        return x, None, None, None


class EncoderConv4NormalizeChannelsFanInFanOutNoKL(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()

        self.latent_dim = latent_dim_conv

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, latent_dim_conv, 3, stride=1, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv4:
                fan = fan_in + conv.out_channels  # fan_out = C_conv4 for conv4
            elif conv is self.conv1:
                fan = fan_in
            else:
                fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def normalize_channels(self, x):
        norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True) + 1e-8
        return x / norm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv4(x)
        x = self.normalize_channels(F.max_pool2d(x, 2, 2))
        # No ReLU after last conv
        x = x.view(x.size(0), -1)
        return x, None, None, None


class EncoderConvAttentionAfterLastPoolCNormalize(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()

        self.latent_dim = latent_dim_conv

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, latent_dim_conv, 3, stride=1, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv4:
                fan = fan_in + conv.out_channels  # fan_out = C_conv4 for conv4
            elif conv is self.conv1:
                fan = fan_in
            else:
                fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def apply_attention(self, x):
        batch, n, h, w = x.shape
        d = h * w
        x_flat = x.view(batch, n, d)
        scores = torch.matmul((-x_flat), x_flat.transpose(-2, -1)) / (d * n)**0.5
        s = scores.sum(dim=-1)
        sigma = F.softmax(s, dim=-1)
        kFinal = sigma.unsqueeze(-1).unsqueeze(-1) * x
        return kFinal

    def normalize_channels(self, x):
        norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True) + 1e-8
        return x / norm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.apply_attention(x)
        x = self.normalize_channels(x)
        # No ReLU after last conv
        x = x.view(x.size(0), -1)
        return x, None, None, None



class EncoderConvAttentionAfterLastPoolCNormalizeOrder(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()

        self.latent_dim = latent_dim_conv

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, latent_dim_conv, 3, stride=1, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv4:
                fan = fan_in + conv.out_channels  # fan_out = C_conv4 for conv4
            elif conv is self.conv1:
                fan = fan_in
            else:
                fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def apply_attention(self, x):
        batch, n, h, w = x.shape
        d = h * w
        x_flat = x.view(batch, n, d)
        scores = torch.matmul((-x_flat), x_flat.transpose(-2, -1)) / (d * n)**0.5
        s = scores.sum(dim=-1)
        sigma = F.softmax(s, dim=-1)
        kFinal = sigma.unsqueeze(-1).unsqueeze(-1) * x
        return kFinal

    def normalize_channels(self, x):
        norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True) + 1e-8
        return x / norm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.normalize_channels(x)
        x = self.apply_attention(x)
        # No ReLU after last conv
        x = x.view(x.size(0), -1)
        return x, None, None, None




class EncoderConvAttentionAfterLastPoolCNormalizeChannelNorm(nn.Module):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super().__init__()

        self.latent_dim = latent_dim_conv

        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, latent_dim_conv, 3, stride=1, padding=1)

        self.reset_parameters()

    def reset_parameters(self):
        convs = [self.conv1, self.conv2, self.conv3, self.conv4]
        gain = nn.init.calculate_gain('relu')
        for conv in convs:
            if conv.bias is not None:
                conv.bias.data.fill_(0)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            if conv is self.conv4:
                fan = fan_in + conv.out_channels  # fan_out = C_conv4 for conv4
            elif conv is self.conv1:
                fan = fan_in
            else:
                fan = fan_in + fan_out
            if fan > 0:
                std = gain / math.sqrt(fan)
                bound = math.sqrt(3.0) * std
                nn.init.uniform_(conv.weight, -bound, bound)

    def apply_attention(self, x):
        batch, n, h, w = x.shape
        d = h * w
        x_flat = x.view(batch, n, d)
        # Normalize each channel by its Euclidean norm
        channel_norms = torch.norm(x_flat, p=2, dim=-1, keepdim=True) + 1e-8
        x_flat_normalized = x_flat / channel_norms
        scores = torch.matmul((-x_flat_normalized), x_flat_normalized.transpose(-2, -1)) / (d * n)**0.5
        s = scores.sum(dim=-1)
        sigma = F.softmax(s, dim=-1)
        kFinal = sigma.unsqueeze(-1).unsqueeze(-1) * x
        return kFinal

    def normalize_channels(self, x):
        norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True) + 1e-8
        return x / norm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv3(x)
        x = F.relu(F.max_pool2d(x, 2, 2))
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.apply_attention(x)
        x = self.normalize_channels(x)
        # No ReLU after last conv
        x = x.view(x.size(0), -1)
        return x, None, None, None



class EncoderConvAttention(Encoder):
    def apply_attention(self, x):
        batch, n, h, w = x.shape
        d = h * w
        x_flat = x.view(batch, n, d)
        scores = torch.matmul((-x_flat), x_flat.transpose(-2, -1)) / (d * n)**0.5
        s = scores.sum(dim=-1)
        sigma = F.softmax(s, dim=-1)
        kFinal = sigma.unsqueeze(-1).unsqueeze(-1) * x
        return kFinal

    def forward(self, x):
        x = self.conv1(x)
        x = self.apply_attention(x)
        x = F.relu(F.max_pool2d(x, 2, 2))

        x = self.conv2(x)
        x = self.apply_attention(x)
        x = F.relu(F.max_pool2d(x, 2, 2))

        x = self.conv3(x)
        x = self.apply_attention(x)
        x = F.relu(F.max_pool2d(x, 2, 2))

        x = self.conv4(x)
        x = self.apply_attention(x)
        x = F.max_pool2d(x, 2, 2)

        return x


class EncoderConvAttentionKL(Encoder):
    def __init__(self, num_inputs, latent_dim_conv=64):
        super(EncoderConvAttentionKL, self).__init__(num_inputs)
        self.latent_dim = latent_dim_conv
        # Double the last conv to output 2*latent_dim channels for mean and log_var
        self.conv4 = nn.Conv2d(64, 2 * latent_dim_conv, 3, stride=1, padding=1)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)
        self.N = torch.distributions.Normal(0, 1)

    def apply_attention(self, x):
        batch, n, h, w = x.shape
        d = h * w
        x_flat = x.view(batch, n, d)
        scores = torch.matmul(x_flat, x_flat.transpose(-2, -1)) / (d * n)**0.5
        s = scores.sum(dim=-1)
        sigma = F.softmax(s, dim=-1)
        kFinal = sigma.unsqueeze(-1).unsqueeze(-1) * x
        return kFinal

    def forward(self, x):
        x = self.conv1(x)
        x = self.apply_attention(x)
        x = F.relu(F.max_pool2d(x, 2, 2))

        x = self.conv2(x)
        x = self.apply_attention(x)
        x = F.relu(F.max_pool2d(x, 2, 2))

        x = self.conv3(x)
        x = self.apply_attention(x)
        x = F.relu(F.max_pool2d(x, 2, 2))

        x = self.conv4(x)
        x = self.apply_attention(x)
        x = F.max_pool2d(x, 2, 2)  # x shape: (batch, 2*latent_dim, H, W)

        # Split into z_mean and z_log_var
        z_mean, z_log_var = torch.chunk(x, 2, dim=1)

        # Sample z
        noise = self.N.sample(z_mean.shape)
        z = z_mean + torch.exp(z_log_var / 2) * noise

        # KL divergence
        kl = -0.5 * (1 + z_log_var - z_mean**2 - torch.exp(z_log_var)).mean()

        return z, z_mean, z_log_var, kl


class A3Clstm(torch.nn.Module):
    """
    original
    """
    def __init__(self, num_inputs, action_space, args):
        super(A3Clstm, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []
        self.encoder = EncoderWithRelu(num_inputs)

        self.lstm = nn.LSTMCell(1024, self.hidden_size)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx, cx = self.lstm(s, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcMem(torch.nn.Module):
    """ A3C1024fc with running memory on internal feature representations concatenated to critic input """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcMem, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gamma_memory = args.gamma_memory
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []
        self.encoder = EncoderWithRelu(num_inputs)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(1024 + 1024, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)

        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        # Internal memory for running differences on features
        self.running_mem = torch.zeros((1,1024))
        self.prev_x_conv = torch.zeros((1,1024))

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        x_conv = x.view(x.size(0), -1)  # 1024

        if self.monitor_s:
            self.s_values.append(x_conv.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x_actor = F.relu(self.fc(x_conv))  # hidden_size

        # Compute running memory on internal features
        if self.running_mem is None:
            self.running_mem = torch.zeros_like(x_conv)
        if self.prev_x_conv is not None:
            diff = (x_conv - self.prev_x_conv).detach()
            self.running_mem = (diff + self.gamma_memory * self.running_mem).detach()

        self.prev_x_conv = x_conv.detach()

        critic_in = torch.cat([x_conv, self.running_mem], dim=1)

        return self.critic_linear(critic_in), self.actor_linear(x_actor), hx, cx


class A3C1024fc(torch.nn.Module):
    """ experimantal """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fc, self).__init__()
        self.hidden_size = args.hidden_size
        self.monitor_s = getattr(args, 'monitor_s', False)
        if self.monitor_s:
            self.s_values = []
        self.encoder = EncoderWithRelu(num_inputs)

#         self.lstm = nn.LSTMCell(1024, self.hidden_size)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)

        torch.nn.init.kaiming_uniform_(self.fc.weight,nonlinearity="relu")

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4) : (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

#         hx, cx = self.lstm(s, (hx, cx))
#         x = hx
        hx=torch.Tensor([0])
        cx=torch.Tensor([0])

        x = F.relu(self.fc(s))

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcLN(A3C1024fc):
    """ A3C1024fc with trainable LayerNorm applied before conv layers """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcLN, self).__init__(num_inputs, action_space, args)
        self.input_norm = nn.LayerNorm([num_inputs, 80, 80])

    def forward(self, inputs, hx, cx, mem=None):
        inputs = self.input_norm(inputs)  # Apply trainable LayerNorm before conv layers

        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcLN_Output(A3C1024fc):
    """ A3C1024fc with trainable LayerNorm before conv layers and after fc (hidden_size dim) """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcLN_Output, self).__init__(num_inputs, action_space, args)
        self.input_norm = nn.LayerNorm([num_inputs, 80, 80])
        self.output_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, inputs, hx, cx, mem=None):
        inputs = self.input_norm(inputs)  # Apply trainable LayerNorm before conv layers

        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))  # hidden_size
        x = self.output_norm(x)  # Apply trainable LayerNorm after fc

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcNoLastConvRelu(A3C1024fc):
    """ A3C1024fc without relu after last conv layer """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcNoLastConvRelu, self).__init__(num_inputs, action_space, args)
        self.encoder = EncoderNoReluLast(num_inputs)

    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))  # Preserve relu after fc

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcNoLastConvReluLN1stChannel(A3C1024fcNoLastConvRelu):
    """ A3C1024fcNoLastConvRelu with LayerNorm applied only to the 1st input channel """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcNoLastConvReluLN1stChannel, self).__init__(num_inputs, action_space, args)
        self.input_norm_1st = nn.LayerNorm([1, 80, 80])

    def forward(self, inputs, hx, cx, mem=None):
        # Apply LayerNorm only to the 1st input channel
        inputs_1st = inputs[:, 0:1, :, :]  # [batch, 1, 80, 80]
        inputs_rest = inputs[:, 1:, :, :]  # [batch, num_inputs-1, 80, 80]
        inputs_1st = self.input_norm_1st(inputs_1st)
        inputs = torch.cat([inputs_1st, inputs_rest], dim=1)

        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))  # Preserve relu after fc

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcNoLastConvReluSNorm(A3C1024fcNoLastConvRelu):
    """ A3C1024fcNoLastConvRelu with manual mean-std normalization on s """
    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        # Manual mean-std normalization across batch
        mean = torch.mean(s).detach()
        std = torch.std(s).detach()
        s = (s - mean) / (std + 0.000001)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))  # Preserve relu after fc

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcSNorm(A3C1024fc):
    """ A3C1024fc with manual mean-std normalization on fc output """
    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        # Manual mean-std normalization across batch
        mean = torch.mean(s).detach() #.mean(dim=0, keepdim=True)
        std = torch.std(s).detach() #.std(dim=0, keepdim=True)
        s = (s - mean) / (std+0.000001)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))  # hidden_size

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcNoLastConvReluWithDecoder(A3C1024fcNoLastConvRelu):
    """ A3C1024fcNoLastConvRelu with decoder that restores original x from s """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcNoLastConvReluWithDecoder, self).__init__(num_inputs, action_space, args)
        opt = {"latent_dim": 1024}
        device = getattr(args, 'device', torch.device('cpu'))
        self.Decoder = Decoder_AE_nobn(opt, device)

    def forward(self, inputs, hx, cx, mem=None):
        x = self.encoder(inputs)

        s = x.view(x.size(0), -1)

        if self.monitor_s:
            self.s_values.append(s.detach().cpu())

        x_restored = self.Decoder(s)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s))  # Preserve relu after fc

        return self.critic_linear(x), self.actor_linear(x), hx, cx, x_restored


class A3C1024fcVAEWithDecoder(A3C1024fcNoLastConvRelu):
    """ A3C1024fc with VAE encoder and decoder """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcVAEWithDecoder, self).__init__(num_inputs, action_space, args)
        self.encoder = EncoderVAE(num_inputs)
        opt = {"latent_dim": 1024}
        device = getattr(args, 'device', torch.device('cpu'))
        self.Decoder = Decoder_AE_nobn(opt, device)

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        if self.monitor_s:
            self.s_values.append(z.detach().cpu())

        x_restored = self.Decoder(z)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(z))  # Use z instead of s

        return self.critic_linear(x), self.actor_linear(x), hx, cx, x_restored, kl


class A3C1024fcConvKL(A3C1024fcNoLastConvRelu):
    """ A3C1024fc with EncoderConvKL """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcConvKL, self).__init__(num_inputs, action_space, args)
        self.encoder = EncoderConvKL(num_inputs)
        # Adjust fc for the flattened conv latent space: 64 * 5 * 5 = 1600
        self.fc = nn.Linear(1024, self.hidden_size)
        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)

        if self.monitor_s:
            self.s_values.append(z_flat.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(z_flat))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, kl


class A3C1024fcConvKLTanh(A3C1024fcConvKL):
    """ A3C1024fc with EncoderConvKL using tanh after fc """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcConvKLTanh, self).__init__(num_inputs, action_space, args)
        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="tanh")

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)

        if self.monitor_s:
            self.s_values.append(z_flat.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = torch.tanh(self.fc(z_flat))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, kl


class A3C1024fcConvAttentionKL(A3C1024fcConvKL):
    """ A3C1024fc with EncoderConvAttentionKL """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcConvAttentionKL, self).__init__(num_inputs, action_space, args)
        self.encoder = EncoderConvAttentionKL(num_inputs)


class A3C1024fcConvKLNormSLen(A3C1024fcConvKL):
    """ A3C1024fcConvKL with s vector normalized by its Euclidean length """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcConvKLNormSLen, self).__init__(num_inputs, action_space, args)
        self.encoder = EncoderConvWithFanOut1ForLastConvNoKL(num_inputs, latent_dim_conv=64)
        # self.fc = nn.Linear(self.encoder.latent_dim * 5 * 5, self.hidden_size)
        # torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = 1 + fan_out
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        generator = torch.Generator()
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound, generator=generator)

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)
        s_norm = z_flat / (torch.norm(z_flat, p=2, dim=1, keepdim=True) + 1e-8).detach()

        if self.monitor_s:
            self.s_values.append(s_norm.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s_norm))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, None


class A3C1024fcNormalizeChannelsFanInFanOut(A3C1024fcConvKL):
    """ A3C1024fc with EncoderConvNormalizeChannelsFanInFanOutNoKL, s vector normalized by its Euclidean length without detaching norm, and fan_in for fc set to encoder out num channels """
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.encoder = EncoderConv4NormalizeChannelsFanInFanOutNoKL(num_inputs, latent_dim_conv=64)
        # Custom initialization
        fan_in = self.encoder.latent_dim  # num out channels of previous (encoder)
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = fan_in + fan_out
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)
        # norm = torch.norm(z_flat, p=2, dim=1, keepdim=True) + 1e-8
        s_norm = z_flat # / norm  # No detach

        if self.monitor_s:
            self.s_values.append(s_norm.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s_norm))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, None


class A3C1024fcConvAttentionAfterLastPoolCNormalize(A3C1024fcConvKL):
    """ A3C1024fc with EncoderConvAttentionAfterLastPoolCNormalize """
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.encoder = EncoderConvAttentionAfterLastPoolCNormalize(num_inputs, latent_dim_conv=64)
        # Custom initialization
        fan_in = self.encoder.latent_dim  # num out channels of previous (encoder)
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = fan_in + fan_out
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)
        s_norm = z_flat

        if self.monitor_s:
            self.s_values.append(s_norm.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s_norm))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, None


class A3C1024fcConvAttentionAfterLastPoolCNormalizeChannelNorm(A3C1024fcConvKL):
    """ A3C1024fc with EncoderConvAttentionAfterLastPoolCNormalizeChannelNorm """
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.encoder = EncoderConvAttentionAfterLastPoolCNormalizeChannelNorm(num_inputs, latent_dim_conv=64)
        # Custom initialization
        fan_in = self.encoder.latent_dim  # num out channels of previous (encoder)
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = fan_in + fan_out
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)
        s_norm = z_flat

        if self.monitor_s:
            self.s_values.append(s_norm.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s_norm))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, None
    


class A3C1024fcConvAttentionAfterLastPoolCNormalizeOrder(A3C1024fcConvKL):
    """ A3C1024fc with EncoderConvAttentionAfterLastPoolCNormalizeOrder """
    def __init__(self, num_inputs, action_space, args):
        super().__init__(num_inputs, action_space, args)
        self.encoder = EncoderConvAttentionAfterLastPoolCNormalizeOrder(num_inputs, latent_dim_conv=64)
        # Custom initialization
        fan_in = self.encoder.latent_dim  # num out channels of previous (encoder)
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
        fan = fan_in + fan_out
        gain = nn.init.calculate_gain("relu")
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)

    def forward(self, inputs, hx, cx, mem=None):
        z, z_mean, z_log_var, kl = self.encoder(inputs)

        z_flat = z.view(z.size(0), -1)
        s_norm = z_flat

        if self.monitor_s:
            self.s_values.append(s_norm.detach().cpu())

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(s_norm))

        return self.critic_linear(x), self.actor_linear(x), hx, cx, None, None
