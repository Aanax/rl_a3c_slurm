"""Standalone Agent.action_train autograd smoke for legacy oracle intrinsic models."""

from pathlib import Path
from types import SimpleNamespace
import runpy
import sys

import numpy as np
import torch

import model
from player_util import Agent


TOKENS = (
    "oracle_intrinsic_nomemory_16_advantage_restoration_0_v1",
    "oracle_intrinsic_concat_diff_16_advantage_restoration_0_v1",
    "oracle_intrinsic_nomemory_16_advantage_v1",
    "oracle_intrinsic_concat_diff_16_advantage_v1",
)


class FakeEnv:
    action_space = SimpleNamespace(n=6)

    def __init__(self):
        self.steps = 0

    def step(self, action):
        del action
        self.steps += 1
        return (
            np.full((1, 80, 80), self.steps / 10, dtype=np.float32),
            1.0,
            self.steps == 2,
            {},
        )


def load_args(repo, token):
    main_path = repo / "src" / "main.py"
    config_path = repo / "configs" / ("run_config_" + token + ".ini")
    original_argv = sys.argv
    try:
        sys.argv = [str(main_path), str(config_path), "1"]
        return runpy.run_path(str(main_path), run_name="oracle_smoke_config")["args"]
    finally:
        sys.argv = original_argv


def assert_finite_grad_tensor(tensor):
    assert tensor.requires_grad
    assert torch.isfinite(tensor).all()


def smoke(token, repo):
    args = load_args(repo, token)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    env = FakeEnv()
    net = getattr(model, args.model_type)(1, env.action_space, args).train()
    agent = Agent(net, env, args, torch.zeros(1, 80, 80))
    agent.hx = torch.zeros(1, args.hidden_size)
    agent.cx = torch.zeros(1, args.hidden_size)

    assert not getattr(net, "predicts_shared_diff", False)
    agent.action_train()
    agent.action_train()

    assert env.steps == 2
    assert not agent.shared_states
    for values in (agent.values, agent.log_probs, agent.values_intrinsic, agent.x_restoreds):
        assert len(values) == 2
    assert len(agent.states) == 2
    assert len(agent.next_states) == 2
    for log_prob, value, intrinsic_value, restored in zip(
        agent.log_probs, agent.values, agent.values_intrinsic, agent.x_restoreds
    ):
        assert restored.shape == (1, 1, 80, 80)
        for tensor in (log_prob, value, intrinsic_value, restored):
            assert_finite_grad_tensor(tensor)

    loss = sum(
        -log_prob.mean()
        + value.pow(2).mean()
        + intrinsic_value.pow(2).mean()
        + restored.pow(2).mean()
        for log_prob, value, intrinsic_value, restored in zip(
            agent.log_probs, agent.values, agent.values_intrinsic, agent.x_restoreds
        )
    )
    loss.backward()
    grads = [parameter.grad for parameter in net.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert any(torch.any(grad != 0) for grad in grads)
    print(
        "PASS {} actor_input_mode={} critic_input_mode={} gamma_restoration={}".format(
            token,
            args.actor_input_mode,
            args.critic_input_mode,
            args.gamma_restoration,
        )
    )


def main():
    repo = Path(__file__).resolve().parents[1]
    for token in TOKENS:
        smoke(token, repo)
    print("4 passed")


if __name__ == "__main__":
    main()
