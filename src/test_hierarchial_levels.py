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


def _args(**kwargs):
    base = dict(
        hidden_size=64, monitor_s=False, use_rmsnorm=False, num_options=4
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_named_output_and_persistence():
    args = _args()
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
    assert out[10] is out.beta2
    assert out[11] is out.terminated1
    assert out[12] is out.terminated2
    assert m.level1.beta is None
    assert m.level1.use_beta is False
    assert m.level1.critic_int is None
    assert out.V1_int is None

    opt_after = m.level2.current_action.clone()
    m.level2.beta.bias.data.fill_(-50.0)
    _ = m(x, None, None)
    assert torch.equal(m.level2.current_action, opt_after)
    print('test_named_output_and_persistence passed')


def test_upper_options_dim():
    enc = model.EncoderRules234_2()
    lvl = Level(enc, feat_dim=32 * 4 * 4, n_actions=3, upper_options_dim=0)
    assert lvl.upper_options_dim == 0
    s2 = lvl.encode(torch.randn(1, 64, 4, 4))
    out = lvl.forward_heads(s2)
    assert out.logits.shape == (1, 3)
    print('test_upper_options_dim passed')


def test_internal_critic_head_optional():
    x = torch.randn(1, 2, 80, 80)
    m_off = model.Hierarchial_levels(2, FakeSpace(6), _args())
    assert m_off.use_internal_critic is False
    assert m_off.level1.critic_int is None
    out_off = m_off(x, None, None)
    assert out_off.V1_int is None

    m_on = model.Hierarchial_levels(
        2, FakeSpace(6), _args(use_internal_critic=True)
    )
    assert m_on.use_internal_critic is True
    assert m_on.level1.critic_int is not None
    out_on = m_on(x, None, None)
    assert out_on.V1_int is not None
    assert out_on.V1_int.shape == (1, 1)
    print('test_internal_critic_head_optional passed')


def test_actor_delta_assignments():
    from train import actor_td_weights
    delta = torch.tensor([[1.0]])
    delta2 = torch.tensor([[2.0]])
    delta_int = torch.tensor([[0.5]])
    actor1, actor2 = actor_td_weights(True, delta, delta2, delta_int)
    assert torch.equal(actor1, delta + delta_int)
    assert torch.equal(actor2, delta2)
    actor1, actor2 = actor_td_weights(False, delta, delta2, delta_int)
    assert torch.equal(actor1, delta + delta2)
    assert torch.equal(actor2, actor1)
    print('test_actor_delta_assignments passed')


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
    test_internal_critic_head_optional()
    # train import may fail without setproctitle; skip gracefully
    try:
        test_running_return_td()
        test_actor_delta_assignments()
    except ImportError as e:
        print('skip train-dependent tests:', e)
    print('All smoke tests passed')
