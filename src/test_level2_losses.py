"""Tests for the split level-2 losses: pi trains via CE, beta via advantage."""
from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from train import (
    level2_pi_wave,
    level2_choice_weight,
    level2_policy_ce,
    level2_beta_loss,
    sampled_action_target,
)

LOGITS = [[0.0, 1.0, 2.0]]


def build(beta_val, prev_idx):
    """Level-2 quantities for one step; beta comes from a trainable logit."""
    logits = torch.tensor(LOGITS, requires_grad=True)
    beta_logit = torch.tensor(
        [[torch.logit(torch.tensor(beta_val)).item()]], requires_grad=True
    )
    beta = torch.sigmoid(beta_logit)
    pi = F.softmax(logits, dim=1)
    iota = (
        None if prev_idx is None
        else sampled_action_target(torch.tensor([[prev_idx]]), logits)
    )
    pi_wave = level2_pi_wave(pi, beta, iota)
    return logits, beta_logit, beta, pi, iota, pi_wave


def test_pi_wave_is_the_executed_distribution():
    _, _, _, pi, iota, pi_wave = build(0.25, prev_idx=2)
    assert torch.allclose(pi_wave.sum(dim=1), torch.ones(1))
    # staying on option 2 adds (1 - beta) of mass on top of beta * pi
    assert torch.allclose(pi_wave[0, 2], 0.75 + 0.25 * pi[0, 2])
    assert torch.allclose(pi_wave[0, 0], 0.25 * pi[0, 0])


def test_choice_weight_is_a_probability():
    for beta_val in (0.001, 0.01, 0.1, 0.5, 0.9, 0.999):
        for prev_idx in range(3):
            _, _, beta, pi, iota, pi_wave = build(beta_val, prev_idx)
            weight = level2_choice_weight(pi, pi_wave, beta, iota)
            assert weight.min() >= 0.0
            assert weight.max() <= 1.0 + 1e-5


def test_choice_weight_separates_the_two_timescales():
    """A real option switch counts fully; mere persistence barely counts."""
    _, _, beta, pi, iota, pi_wave = build(0.1, prev_idx=0)
    switched = level2_choice_weight(pi, pi_wave, beta, iota)[0, 2]
    assert torch.allclose(switched, torch.tensor(1.0), atol=1e-4)

    _, _, beta, pi, iota, pi_wave = build(0.1, prev_idx=2)
    continued = level2_choice_weight(pi, pi_wave, beta, iota)[0, 2]
    assert continued < 0.1


def test_choice_weight_is_one_without_a_previous_option():
    _, _, beta, pi, iota, pi_wave = build(0.1, prev_idx=None)
    weight = level2_choice_weight(pi, pi_wave, beta, iota)
    assert torch.allclose(weight, torch.ones_like(weight))


def test_policy_ce_does_not_touch_beta():
    logits, beta_logit, beta, pi, iota, pi_wave = build(0.1, prev_idx=2)
    weight = level2_choice_weight(pi, pi_wave, beta, iota)
    target = sampled_action_target(torch.tensor([[2]]), logits)
    level2_policy_ce(target, pi, weight).backward()
    assert logits.grad is not None
    assert beta_logit.grad is None


def test_beta_loss_does_not_touch_pi():
    logits, beta_logit, beta, pi, iota, pi_wave = build(0.1, prev_idx=2)
    level2_beta_loss(
        pi, pi_wave, beta, iota, torch.tensor([[2]]), torch.tensor([[1.0]])
    ).backward()
    assert beta_logit.grad is not None
    assert logits.grad is None


def beta_grad(beta_val, prev_idx, taken_idx, advantage):
    _, beta_logit, beta, pi, iota, pi_wave = build(beta_val, prev_idx)
    level2_beta_loss(
        pi, pi_wave, beta, iota,
        torch.tensor([[taken_idx]]), torch.tensor([[advantage]]),
    ).backward()
    return beta_logit.grad.item()


def test_beta_moves_the_right_way():
    # positive gradient on a loss means the optimizer pushes beta down
    assert beta_grad(0.1, prev_idx=2, taken_idx=2, advantage=1.0) > 0
    assert beta_grad(0.1, prev_idx=2, taken_idx=2, advantage=-1.0) < 0
    assert beta_grad(0.1, prev_idx=0, taken_idx=2, advantage=1.0) < 0
    assert beta_grad(0.1, prev_idx=0, taken_idx=2, advantage=-1.0) > 0


def test_beta_gradient_is_bounded_by_the_advantage():
    """The 1/beta in the switch case is cancelled by the sigmoid Jacobian."""
    for beta_val in (0.001, 0.01, 0.1, 0.5, 0.9):
        for prev_idx in range(3):
            for advantage in (1.0, -1.0):
                g = beta_grad(beta_val, prev_idx, 2, advantage)
                assert abs(g) <= abs(advantage) + 1e-5, (beta_val, prev_idx, g)


def test_switch_gradient_matches_closed_form():
    for beta_val in (0.01, 0.1, 0.5):
        g = beta_grad(beta_val, prev_idx=0, taken_idx=2, advantage=1.0)
        assert abs(g - (-(1.0 - beta_val))) < 1e-3


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn()
            print(f"{name}: ok")
