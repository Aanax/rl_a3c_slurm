import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import player_util
from player_util import Agent


class _Env:
    def __init__(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return np.zeros((4, 80, 80), dtype=np.float32), 1.0, False, {}


class _LegacyModel:
    def __call__(self, state, hx, cx, mem=None):
        return (
            torch.zeros(1, 1),
            torch.zeros(1, 3),
            torch.Tensor([0]),
            torch.Tensor([0]),
            torch.zeros(1, 64, 4, 4),
        )


class _FutureModel(_LegacyModel):
    predicts_shared_diff = True

    def __init__(self):
        super().__init__()
        self.target = torch.ones(1, 64, 4, 4, requires_grad=True)

    def __call__(self, state, hx, cx, mem=None):
        return super().__call__(state, hx, cx, mem) + (self.target,)


class AgentOptionalOracleCapabilityTest(unittest.TestCase):
    def test_action_train_uses_capability_flag(self):
        for name, model, expected_states in (
            ('legacy', _LegacyModel(), 0),
            ('future', _FutureModel(), 1),
        ):
            with self.subTest(name=name):
                env = _Env()
                agent = Agent(
                    model,
                    env,
                    SimpleNamespace(hidden_size=1),
                    torch.zeros(4, 80, 80),
                )
                with patch.object(player_util, 'model_module', SimpleNamespace()):
                    agent.action_train()

                self.assertEqual(env.steps, 1)
                self.assertEqual(len(agent.x_restoreds), 1)
                self.assertEqual(len(agent.shared_states), expected_states)
                if expected_states:
                    self.assertTrue(torch.equal(agent.shared_states[0], model.target))
                    self.assertFalse(agent.shared_states[0].requires_grad)


if __name__ == '__main__':
    unittest.main()
