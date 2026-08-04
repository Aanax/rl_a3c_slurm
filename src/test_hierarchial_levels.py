"""Smoke tests for Hierarchial_levels named output + sticky persistence."""
from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from types import SimpleNamespace

import model
from level import Level


class FakeSpace(object):
    def __init__(self, n):
        self.n = n


def test_named_output_and_persistence():
    args = SimpleNamespace(
        hidden_size=64, monitor_s=False, use_rmsnorm=False, num_options=4
    )
    m = model.Hierarchial_levels(2, FakeSpace(6), args)
    x = torch.randn(1, 2, 80, 80)
    out = m(x, None, None)
    assert isinstance(out, model.HierarchialLevelsOutput)
    assert out.V1.shape == (1, 1)
    assert out.a1_logits.shape == (1, 6)
    assert out.V2.shape == (1, 1)
    assert out.a2_logits.shape == (1, 4)
    assert out.a1.shape == (1, 1)
    assert out.a2.shape == (1, 1)
    # named fields match legacy positional indices
    assert out[0] is out.V1
    assert out[6] is out.V2
    assert out[7] is out.a2_logits
    assert out[8] is out.a1
    assert out[9] is out.a2
    assert out[10] is out.beta1
    assert out[11] is out.beta2

    opt_after = m.level2.current_action.clone()
    act_after = m.level1.current_action.clone()
    m.level2.beta.bias.data.fill_(-50.0)
    m.level1.beta.bias.data.fill_(-50.0)
    _ = m(x, None, None)
    assert torch.equal(m.level2.current_action, opt_after)
    assert torch.equal(m.level1.current_action, act_after)
    print('test_named_output_and_persistence passed')


def test_upper_options_dim():
    enc = model.EncoderRules234_2()
    lvl = Level(enc, feat_dim=32 * 4 * 4, n_actions=3, upper_options_dim=0)
    assert lvl.upper_options_dim == 0
    s2 = lvl.encode(torch.randn(1, 64, 4, 4))
    out = lvl.forward_heads(s2)
    assert out.logits.shape == (1, 3)
    print('test_upper_options_dim passed')


def test_running_return_td():
    from train import running_return_td
    R = torch.tensor([[0.0]])
    reward = 1.0
    value = torch.tensor([[0.5]], requires_grad=True)
    R, adv, delta = running_return_td(R, reward, value, gamma=0.0)
    assert abs(R.item() - 1.0) < 1e-6
    assert abs(adv.item() - 0.5) < 1e-6
    assert not delta.requires_grad
    print('test_running_return_td passed')


if __name__ == '__main__':
    test_upper_options_dim()
    test_named_output_and_persistence()
    # train import may fail without setproctitle; skip gracefully
    try:
        test_running_return_td()
    except ImportError as e:
        print('skip test_running_return_td:', e)
    print('All smoke tests passed')
