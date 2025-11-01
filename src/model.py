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
    """ A3C1024fc with running memory on internal feature representations concatenated to critic input """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcMem, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gamma_memory = args.gamma_memory
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        num_outputs = action_space.n
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
        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x_conv = x.view(x.size(0), -1)  # 1024

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


class A3C1024fcLN(torch.nn.Module):
    """ A3C1024fc with trainable LayerNorm applied before conv layers """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcLN, self).__init__()
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)
        self.input_norm = nn.LayerNorm([num_inputs, 80, 80])

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
        inputs = self.input_norm(inputs)  # Apply trainable LayerNorm before conv layers

        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x = x.view(x.size(0), -1)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(x))

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcLN_Output(torch.nn.Module):
    """ A3C1024fc with trainable LayerNorm before conv layers and after fc (hidden_size dim) """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcLN_Output, self).__init__()
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_outputs)
        self.fc = nn.Linear(1024, self.hidden_size)
        self.input_norm = nn.LayerNorm([num_inputs, 80, 80])
        self.output_norm = nn.LayerNorm(self.hidden_size)

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
        inputs = self.input_norm(inputs)  # Apply trainable LayerNorm before conv layers

        x = F.relu(F.max_pool2d(self.conv1(inputs), 2, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2, 2))

        x = x.view(x.size(0), -1)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(x))  # hidden_size
        x = self.output_norm(x)  # Apply trainable LayerNorm after fc

        return self.critic_linear(x), self.actor_linear(x), hx, cx


class A3C1024fcSNorm(torch.nn.Module):
    """ A3C1024fc with manual mean-std normalization on fc output """
    def __init__(self, num_inputs, action_space, args):
        super(A3C1024fcSNorm, self).__init__()
        self.hidden_size = args.hidden_size
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

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

        torch.nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")

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

        x = x.view(x.size(0), -1)

        # Manual mean-std normalization across batch
        mean = torch.mean(x) #.mean(dim=0, keepdim=True)
        std = torch.std(x) #.std(dim=0, keepdim=True)
        x = (x - mean) / (std+0.000001)

        hx = torch.Tensor([0])
        cx = torch.Tensor([0])

        x = F.relu(self.fc(x))  # hidden_size

        return self.critic_linear(x), self.actor_linear(x), hx, cx
