"""Checks for the two-level oracle model with memory.

Run from the repository root:
    python -m unittest discover tests
"""
import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# gym and setproctitle are only needed on the cluster. Stub them when missing.
if "gym" not in sys.modules:
    try:
        import gym  # noqa: F401
    except ImportError:
        gym_stub = types.ModuleType("gym")
        gym_stub.__version__ = "stub"
        gym_stub.ObservationWrapper = object
        gym_stub.Wrapper = object
        spaces = types.ModuleType("gym.spaces")
        box = types.ModuleType("gym.spaces.box")
        box.Box = object
        sys.modules.update({"gym": gym_stub, "gym.spaces": spaces, "gym.spaces.box": box})
try:
    import setproctitle  # noqa: F401
except ImportError:
    stub = types.ModuleType("setproctitle")
    stub.setproctitle = lambda name: None
    sys.modules["setproctitle"] = stub

import model
import train as train_module
from player_util import Agent
from shared_optim import SharedAdam


NUM_ACTIONS = 6


def make_args(**overrides):
    args = SimpleNamespace(
        experiment_name="unit_test", env="FakeGame", seed=1, total_steps_stop=10**9,
        log_dir="", lr=1e-3, num_steps=8, entropy_coef=0.0,
        gamma=0.875, gamma_oracle=0.0, num_options=2, gamma2=0.984375, gamma_oracle2=0.75,
        w_restoration_loss=1.0, monitor_s_save_interval=40,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeGame:
    """Random 1x80x80 frames, a reward now and then, a lost life every `life_length` steps."""

    def __init__(self, life_length=20):
        self.observation_space = SimpleNamespace(shape=(1, 80, 80))
        self.action_space = SimpleNamespace(n=NUM_ACTIONS)
        self.life_length = life_length
        self.t = 0
        self.rng = np.random.RandomState(0)

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.t = 0
        return self.rng.randn(1, 80, 80).astype(np.float32)

    def step(self, action):
        self.t += 1
        reward = 1.0 if self.rng.rand() < 0.1 else 0.0
        done = self.t >= self.life_length
        return self.rng.randn(1, 80, 80).astype(np.float32), reward, done, {}


def frame():
    return torch.randn(1, 1, 80, 80)


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = model.A3CRules2378OracleTwoLevelMemory(
            1, SimpleNamespace(n=NUM_ACTIONS), make_args(num_options=3)
        )

    def test_output_shapes(self):
        out = self.net(frame(), None)
        self.assertEqual(out['action_logits'].shape, (1, NUM_ACTIONS))
        self.assertEqual(out['option_logits'].shape, (1, 3))
        for key in ['value', 'value_intrinsic', 'value2', 'value_intrinsic2']:
            self.assertEqual(out[key].shape, (1, 1), key)
        self.assertEqual(out['option'].shape, (1, 1))
        self.assertEqual(out['s1'].shape, (1, 64, 4, 4))
        self.assertEqual(out['s2'].shape, (1, 32, 4, 4))
        self.assertEqual(out['oracle1_prediction'].shape, (1, 64, 4, 4))
        self.assertEqual(out['oracle2_prediction'].shape, (1, 32, 4, 4))

    def test_no_memory_equals_zero_change(self):
        x = frame()
        s1 = self.net(x, None, option=1)['s1']
        a = self.net(x, None, option=1)
        b = self.net(x, s1, option=1)   # previous s1 == current s1 -> change is zero
        self.assertTrue(torch.allclose(a['value'], b['value']))
        self.assertTrue(torch.allclose(a['action_logits'], b['action_logits']))

    def test_memory_changes_level1_but_not_level2(self):
        x = frame()
        a = self.net(x, None, option=0)
        b = self.net(x, torch.randn(1, 64, 4, 4), option=0)
        self.assertFalse(torch.allclose(a['value'], b['value']))
        self.assertFalse(torch.allclose(a['oracle1_prediction'], b['oracle1_prediction']))
        self.assertTrue(torch.allclose(a['option_logits'], b['option_logits']))
        self.assertTrue(torch.allclose(a['value2'], b['value2']))

    def test_option_is_used_and_changes_level1(self):
        x = frame()
        a = self.net(x, None, option=0)
        b = self.net(x, None, option=2)
        self.assertEqual(a['option'].item(), 0)
        self.assertEqual(b['option'].item(), 2)
        self.assertFalse(torch.allclose(a['action_logits'], b['action_logits']))

    def test_eval_mode_picks_most_likely_option(self):
        self.net.eval()
        x = frame()
        out = self.net(x, None)
        self.assertEqual(out['option'].item(), out['option_logits'].argmax().item())

    def test_memory_gradient_reaches_previous_step(self):
        x1, x2 = frame(), frame()
        first = self.net(x1, None, option=0)
        second = self.net(x2, first['s1'], option=0)
        second['value'].sum().backward()
        self.assertIsNotNone(self.net.shared_encoder.conv1.weight.grad)


class OptionReturnsTest(unittest.TestCase):
    def test_hand_example(self):
        # Steps (forward in time): option 0, 0, 1. Rewards 1, 0, 2. gamma 0.5.
        # Option 0 ended after step 1 with value 10. Batch tail value is 4.
        gamma = 0.5
        returns = [None, None]
        r_step2 = train_module.update_option_returns(returns, 1, 2.0, gamma, torch.tensor(4.0))
        self.assertAlmostEqual(r_step2.item(), 0.5 * 4 + 2)
        r_step1 = train_module.update_option_returns(returns, 0, 0.0, gamma, torch.tensor(10.0))
        self.assertAlmostEqual(r_step1.item(), 0.5 * 10 + 0)
        self.assertAlmostEqual(returns[1].item(), 0.5 * (0.5 * 4 + 2))
        r_step0 = train_module.update_option_returns(returns, 0, 1.0, gamma, None)
        self.assertAlmostEqual(r_step0.item(), 0.5 * 5 + 1)


class AgentTest(unittest.TestCase):
    def test_memory_and_storage(self):
        torch.manual_seed(0)
        args = make_args(num_options=2)
        env = FakeGame(life_length=5)
        player = Agent(None, env, args, None)
        player.model = model.A3CRules2378OracleTwoLevelMemory(1, env.action_space, args)
        player.state = torch.from_numpy(env.reset()).float()

        player.action_train()
        self.assertIsNotNone(player.previous_s1)
        for _ in range(4):
            player.action_train()
        self.assertTrue(player.done)
        self.assertIsNone(player.previous_s1)

        n = len(player.rewards)
        self.assertEqual(n, 5)
        for name in ['log_probs', 'values', 'values_intrinsic', 'oracle1_predictions',
                     'options', 'log_probs2', 'values2', 'values_intrinsic2',
                     'oracle2_predictions', 'option_end_values']:
            self.assertEqual(len(getattr(player, name)), n, name)
        # one extra (zero) state after the lost life
        self.assertEqual(len(player.s1_states), n + 1)
        self.assertTrue(torch.all(player.s1_states[-1] == 0))
        # option-end values are filled exactly where the option changes
        for i in range(n - 1):
            changed = player.options[i].item() != player.options[i + 1].item()
            self.assertEqual(player.option_end_values[i] is not None, changed)
        self.assertIsNone(player.option_end_values[-1])


class TrainLoopTest(unittest.TestCase):
    def test_a_few_batches(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(log_dir=tmp + "/", num_steps=8, total_steps_stop=60)
            env = FakeGame(life_length=20)
            shared_model = model.A3CRules2378OracleTwoLevelMemory(1, env.action_space, args)
            before = [p.detach().clone() for p in shared_model.parameters()]
            optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
            frames_total = SimpleNamespace(value=0)

            original_env = train_module.atari_env
            train_module.atari_env = lambda env_id, env_conf, a: FakeGame(life_length=20)
            try:
                train_module.train(0, args, shared_model, optimizer, {}, frames_total)
            finally:
                train_module.atari_env = original_env

            self.assertGreater(frames_total.value, 60)
            changed = [not torch.equal(b, p) for b, p in zip(before, shared_model.parameters())]
            self.assertTrue(all(changed), "every parameter should get a gradient")

            log_dir = os.path.join(tmp, args.experiment_name)
            with open(os.path.join(log_dir, "losses_rank0.csv")) as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ['batch_num', 'policy_loss', 'value_loss',
                                       'restoration_loss', 'value_intrinsic_loss'])
            self.assertGreater(len(rows), 5)
            for row in rows[1:]:
                self.assertTrue(all(np.isfinite(float(v)) for v in row))
            self.assertTrue(os.path.exists(
                os.path.join(log_dir, "s_monitor_FakeGame_rank0_game1.pkl")))


if __name__ == "__main__":
    unittest.main()
