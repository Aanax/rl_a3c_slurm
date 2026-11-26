"""
Adam optimizer whose statistics live in shared memory, so all training workers
update the same running averages.
"""
from __future__ import division
from math import sqrt
import torch
import torch.optim as optim
from collections import defaultdict


class SharedAdam(optim.Optimizer):
    """Adam with shared state."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0):
        defaults = defaultdict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1
                if group["weight_decay"] != 0:
                    grad = grad.add(p.data, alpha=group["weight_decay"])

                # Running averages of the gradient and of its square
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction for the first steps
                step_t = state["step"].item()
                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                step_size = group["lr"] / bias_correction1
                denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(group["eps"])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return None
