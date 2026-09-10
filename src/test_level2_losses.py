"""Tests for the split level-2 losses: pi trains via sampled score, beta via advantage."""
from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from train import (
    PI_WAVE_EPS,
    level2_pi_wave,
    level2_ext_policy_loss,
    level2_beta_loss,
    level2_entropy_log_prob,
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


def test_ext_policy_loss_is_sampled_score():
    """Loss uses log π(a), π^ext(a), π̃(a), β — not a sum over the simplex."""
    logits, _, beta, pi, iota, pi_wave = build(0.1, prev_idx=2)
    action = torch.tensor([[2]])
    pi_ext = torch.tensor([[0.1, 0.2, 0.7]])
    log_pi_a = F.log_softmax(logits, dim=1).gather(1, action)
    loss = level2_ext_policy_loss(log_pi_a, pi_wave, beta, action, pi_ext)
    expected = -(
        pi_ext[0, 2] * beta / (pi_wave[0, 2] + PI_WAVE_EPS)
    ).detach() * log_pi_a
    assert torch.allclose(loss, expected)


def test_ext_policy_loss_does_not_touch_beta():
    logits, beta_logit, beta, pi, iota, pi_wave = build(0.1, prev_idx=2)
    action = torch.tensor([[2]])
    log_pi_a = F.log_softmax(logits, dim=1).gather(1, action)
    pi_ext = sampled_action_target(action, logits)
    level2_ext_policy_loss(log_pi_a, pi_wave, beta, action, pi_ext).backward()
    assert logits.grad is not None
    assert beta_logit.grad is None


def test_ext_policy_loss_ignores_other_actions():
    """A one-hot π^ext on a different option must not enter the score."""
    logits, _, beta, pi, iota, pi_wave = build(0.1, prev_idx=2)
    action = torch.tensor([[2]])
    log_pi_a = F.log_softmax(logits, dim=1).gather(1, action)
    pi_ext_off = sampled_action_target(torch.tensor([[0]]), logits)
    pi_ext_on = sampled_action_target(action, logits)
    off = level2_ext_policy_loss(log_pi_a, pi_wave, beta, action, pi_ext_off)
    on = level2_ext_policy_loss(log_pi_a, pi_wave, beta, action, pi_ext_on)
    # off-target mass at a is 0, so the coefficient (and the loss) is 0
    assert torch.allclose(off, torch.zeros_like(off))
    assert on.abs() > off.abs()


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


def test_entropy_bonus_is_beta_log_pi():
    """r2 -= entropy_log_prob adds -β log π(a) for the current option."""
    logits, _, beta, _, iota, _ = build(0.25, prev_idx=2)
    log_pi_a = F.log_softmax(logits, dim=1)[:, 2:3]
    term = level2_entropy_log_prob(log_pi_a, beta, iota)
    assert torch.allclose(term, 0.25 * log_pi_a)
    assert not term.requires_grad
    # Fresh sample is always from π, so no β factor.
    term_fresh = level2_entropy_log_prob(log_pi_a, beta, iota=None)
    assert torch.allclose(term_fresh, log_pi_a.detach())


if __name__ == '__main__':
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_'):
            fn()
            print(f"{name}: ok")
