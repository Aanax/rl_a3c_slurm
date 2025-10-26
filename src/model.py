from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init


class A3Clstm(torch.nn.Module):
    """
    original
    """
    def __init__(self, num_inputs, action_space, args):
        super(A3Clstm, self).__init__()
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.lstm = nn.LSTMCell(1024, self.hidden_size)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)

        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2.bias.data.fill_(0)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv3.bias.data.fill_(0)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)
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
        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcMem(torch.nn.Module):
    """ A3C1024fc with running memory concatenated to critic input """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcMem, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        num_outputs = action_space.n
        # Assume mem is same size as input state, flatten it
        state_flat_size = 4 * 80 * 80  # Based on typical Pong/Atari, adjust if needed
        self.mem_proj = nn.Linear(state_flat_size, 1024)
        self.critic_linear = nn.Linear(1024 + 1024, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)

        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2.bias.data.fill_(0)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv3.bias.data.fill_(0)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")
        torch.nn.init.kaiming_uniform_(self.mem_proj.weight, nonlinearity="relu")

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.train()

    def forward(self, inputs, hx, cx, mem=None):
        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x_conv = x.view(x.size(0), -1)  # 1024

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x_actor = F.relu(self.fc(x_conv))  # hidden_size

        mem_in = torch.zeros_like(x_conv) if mem is None else F.relu(self.mem_proj(mem.view(x_conv.size(0), -1)))
        critic_in = torch.cat([x_conv, mem_in], dim=1)

        return self.critic_linear(critic_in), self.actor_linear(x_actor), hx, cx


class A3C1024fcWithMemory(torch.nn.Module):
    """Same as A3C1024fc but adds running memory concatenated to critic input"""
    def __init__(self, num_inputs, action_space, args, mem_size=None):
        super(A3C1024fcWithMemory, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_inputs = num_inputs
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        num_outputs = action_space.n
        # Define mem_fc with flattened state size
        state_size = num_inputs * 80 * 80  # Change if needed
        self.mem_fc = nn.Linear(state_size, self.hidden_size)
        self.critic_linear = nn.Linear(self.hidden_size * 2, 1)
        self.actor_linear = nn.Linear(self.hidden_size * 2, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)

        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2.bias.data.fill_(0)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv3.bias.data.fill_(0)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)

        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")
        torch.nn.init.kaiming_uniform_(self.mem_fc.weight, nonlinearity="relu")

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
        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x = x.view(x.size(0), -1)
        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(x))

        mem_processed = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        if mem is not None:
            mem_processed = F.relu(self.mem_fc(mem))

        x_aug = torch.cat([x, mem_processed], dim=1)

        return self.critic_linear(x_aug), self.actor_linear(x_aug), hx, cx

class A3C1024fc(torch.nn.Module):
    """ experimantal """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fc, self).__init__()
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

#         self.lstm = nn.LSTMCell(1024, self.hidden_size)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)


        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.conv1.bias.data.fill_(0)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv2.bias.data.fill_(0)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv3.bias.data.fill_(0)
        self.conv4.weight.data.mul_(relu_gain)
        self.conv4.bias.data.fill_(0)

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
        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x = x.view(x.size(0), -1)

#         hx, cx = self.lstm(x, (hx, cx))
#         x = hx
        hx=torch.Tensor([0])
        cx=torch.Tensor([0])

        x = F.relu(self.fc(x))

        return self.critic_linear(x), self.actor_linear(x), hx, cx
