import ast
import os
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model
from player_util import Agent
from model import (
    _FutureSharedDiffTargetMixin,
    A3CRules2378OracleNoSplitSharedFutureDiff,
    A3CRules2378OracleNoSplitSharedFutureDiffIntrinsicCritic,
)


class _ActionSpace:
    n = 3


class _TerminalEnv:
    def step(self, action):
        return np.zeros((4, 80, 80), dtype=np.float32), 1.0, True, {}


class _TargetModel(_FutureSharedDiffTargetMixin):
    predicts_shared_diff = True

    def __init__(self):
        self.prev_shared = None
        self.target_seen_before_reset = False
        self.state_target = torch.zeros(1, 64, 4, 4)
        self.next_target = torch.ones(1, 64, 4, 4)

    def __call__(self, state, hx, cx, mem=None):
        self.raw_state = state
        self.prev_shared = object()
        return (
            torch.zeros(1, 1), torch.zeros(1, 3), torch.Tensor([0]),
            torch.Tensor([0]), torch.zeros(1, 64, 4, 4), self.state_target,
        )

    def terminal_next_shared(self, next_state):
        self.raw_next_state = next_state
        self.target_seen_before_reset = self.prev_shared is not None
        return self.next_target


class _LegacyOracleModel:
    predicts_shared_diff = False

    def __call__(self, state, hx, cx, mem=None):
        return (
            torch.zeros(1, 1), torch.zeros(1, 3), torch.Tensor([0]),
            torch.Tensor([0]), torch.zeros(1, 64, 4, 4),
        )


class _TwoStepEnv:
    def step(self, action):
        return np.zeros((4, 80, 80), dtype=np.float32), 0.0, False, {}


class _TerminalSecondEnv:
    def __init__(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return np.zeros((4, 80, 80), dtype=np.float32), 0.0, self.steps == 2, {}


class _SequenceTargetModel(_FutureSharedDiffTargetMixin):
    def __init__(self):
        self.forward_calls = 0
        self.terminal_calls = 0

    def __call__(self, state, hx, cx, mem=None):
        self.forward_calls += 1
        shared = torch.full((1, 64, 4, 4), float(self.forward_calls))
        return (
            torch.zeros(1, 1), torch.zeros(1, 3), torch.Tensor([0]),
            torch.Tensor([0]), torch.zeros(1, 64, 4, 4), shared,
        )

    def terminal_next_shared(self, state):
        self.terminal_calls += 1
        return torch.full((1, 64, 4, 4), float(self.forward_calls + 1))


class SharedFutureDeltaOracleTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(monitor_s=False, use_rmsnorm=False)

    def _assert_oracle_contract(self, model, intrinsic=False):
        current = torch.randn(2, 4, 80, 80)
        future = torch.randn(2, 4, 80, 80)
        with torch.no_grad():
            model.shared_future_delta_head.weight.zero_()
            model.shared_future_delta_head.bias.fill_(-1)
        output = model(current, None, None)
        prediction = output[4]
        shared_t = output[-1]

        self.assertEqual(model.shared_future_delta_head.kernel_size, (1, 1))
        self.assertEqual(prediction.shape, (2, 64, 4, 4))
        self.assertTrue(model.predicts_shared_diff)
        self.assertTrue((prediction < 0).all())
        self.assertEqual(len(output), 6 + int(intrinsic))
        self.assertEqual(shared_t.shape, prediction.shape)
        self.assertFalse(shared_t.requires_grad)

        before = model.prev_shared
        next_shared = model.terminal_next_shared(future)
        self.assertTrue(torch.equal(before.detach(), shared_t))
        self.assertFalse(next_shared.requires_grad)
        self.assertEqual(shared_t.shape, prediction.shape)
        self.assertEqual(next_shared.shape, prediction.shape)
        self.assertIs(model.prev_shared, before)

    def test_shared_future_delta_output_and_detached_targets(self):
        self._assert_oracle_contract(
            A3CRules2378OracleNoSplitSharedFutureDiff(4, _ActionSpace(), self._args())
        )

    def test_shared_future_delta_intrinsic_output(self):
        self._assert_oracle_contract(
            A3CRules2378OracleNoSplitSharedFutureDiffIntrinsicCritic(
                4, _ActionSpace(), self._args()
            ),
            intrinsic=True,
        )

    def test_agent_keeps_terminal_feature_transition_before_reset(self):
        model = _TargetModel()
        agent = Agent(
            model,
            _TerminalEnv(),
            SimpleNamespace(hidden_size=1),
            torch.zeros(4, 80, 80),
        )

        agent.action_train()

        self.assertTrue(model.target_seen_before_reset)
        self.assertTrue(torch.equal(agent.states[0], model.raw_state))
        self.assertTrue(torch.equal(agent.next_states[0], model.raw_next_state))
        self.assertEqual(len(agent.shared_states), 2)
        self.assertTrue(torch.equal(agent.shared_states[0], model.state_target))
        self.assertFalse(agent.shared_states[0].requires_grad)
        self.assertTrue(torch.equal(agent.shared_states[1], model.next_target))
        self.assertFalse(agent.shared_states[1].requires_grad)
        self.assertEqual(len(agent.x_restoreds), 1)
        self.assertIsNone(model.prev_shared)
        agent.clear_actions()
        self.assertEqual(agent.shared_states, [])
        self.assertEqual(agent.x_restoreds, [])

        legacy_agent = Agent(
            _LegacyOracleModel(), _TerminalEnv(), SimpleNamespace(hidden_size=1),
            torch.zeros(4, 80, 80),
        )
        legacy_agent.action_train()
        self.assertEqual(len(legacy_agent.x_restoreds), 1)
        self.assertEqual(legacy_agent.shared_states, [])

    def test_nonterminal_shared_transitions_use_bootstrap_output(self):
        sequence_model = _SequenceTargetModel()
        agent = Agent(
            sequence_model, _TwoStepEnv(), SimpleNamespace(hidden_size=1),
            torch.zeros(4, 80, 80),
        )
        agent.action_train()
        agent.action_train()
        self.assertEqual([state[0, 0, 0, 0].item() for state in agent.shared_states], [1.0, 2.0])
        self.assertEqual(sequence_model.terminal_calls, 0)

        train_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py')
        with open(train_path) as train_file:
            tree = ast.parse(train_file.read())
        bootstrap = next(
            node for node in ast.walk(tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Attribute)
            and node.test.operand.attr == 'done'
            and any(isinstance(child, ast.If) for child in node.body)
        )
        block = compile(ast.Module(body=bootstrap.body[:3]), train_path, 'exec')
        player = SimpleNamespace(
            done=False, state=agent.state, hx=agent.hx, cx=agent.cx,
            model=sequence_model, shared_states=agent.shared_states,
        )
        scope = {'model': model, 'player': player}
        exec(block, scope)
        self.assertEqual(len(scope['model_output']), 6)
        self.assertEqual(
            [state[0, 0, 0, 0].item() for state in agent.shared_states],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(sequence_model.forward_calls, 3)
        self.assertEqual(sequence_model.terminal_calls, 0)

        legacy_player = SimpleNamespace(
            done=False, state=agent.state, hx=agent.hx, cx=agent.cx,
            model=_LegacyOracleModel(), shared_states=[],
        )
        exec(block, {'model': model, 'player': legacy_player})
        self.assertEqual(legacy_player.shared_states, [])

    def test_terminal_shared_transition_uses_terminal_encoder_once(self):
        sequence_model = _SequenceTargetModel()
        agent = Agent(
            sequence_model, _TerminalSecondEnv(), SimpleNamespace(hidden_size=1),
            torch.zeros(4, 80, 80),
        )
        agent.action_train()
        agent.action_train()
        self.assertEqual(
            [state[0, 0, 0, 0].item() for state in agent.shared_states],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(sequence_model.forward_calls, 2)
        self.assertEqual(sequence_model.terminal_calls, 1)

    def test_train_uses_discounted_shared_delta_target(self):
        train_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py')
        with open(train_path) as train_file:
            tree = ast.parse(train_file.read())
        target_block = compile(ast.Module(body=[next(
            node for node in ast.walk(tree) if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Name)
            and node.test.func.id == 'isinstance'
            and any(isinstance(child, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'G_t'
                            for target in child.targets)
                    for child in ast.walk(node))
            and any(isinstance(child, ast.Attribute) and child.attr == 'shared_states'
                    for child in ast.walk(node))
        )]), train_path, 'exec')
        player = SimpleNamespace(
            model=_TargetModel(),
            states=[torch.tensor([100.0])],
            next_states=[torch.tensor([200.0])],
            shared_states=[torch.tensor([1.0]), torch.tensor([-2.0])],
            rewards=[0.0],
        )
        scope = {'F': torch.nn.functional, 'model': model,
                 'args': SimpleNamespace(gamma_restoration=0.5, relu_g_const=True),
                 'player': player, 'i': 0, 'G_t': torch.tensor([-99.0])}
        exec(target_block, scope)
        self.assertTrue(torch.equal(scope['G_t'], torch.tensor([-52.5])))

        player.model = _LegacyOracleModel()
        scope = {'F': torch.nn.functional, 'model': model,
                 'args': SimpleNamespace(gamma_restoration=0.5, relu_g_const=False),
                 'player': player, 'i': 0, 'G_t': torch.tensor([-99.0])}
        exec(target_block, scope)
        self.assertTrue(torch.equal(scope['G_t'], torch.tensor([50.5])))

        player.states[0] = torch.tensor([1.0])
        player.next_states[0] = torch.tensor([-2.0])
        scope = {'F': torch.nn.functional, 'model': model,
                 'args': SimpleNamespace(gamma_restoration=0.0, relu_g_const=True),
                 'player': player, 'i': 0, 'G_t': torch.tensor([0.0])}
        exec(target_block, scope)
        self.assertTrue(torch.equal(scope['G_t'], torch.tensor([-3.0])))

    def test_eval_keeps_shared_oracle_prediction(self):
        eval_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'eval.py')
        with open(eval_path) as eval_file:
            tree = ast.parse(eval_file.read())
        collection = next(
            node for node in ast.walk(tree) if isinstance(node, ast.If)
            and any(isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Call)
                    and isinstance(child.value.func, ast.Attribute)
                    and child.value.func.attr == 'append'
                    and isinstance(child.value.func.value, ast.Name)
                    and child.value.func.value.id == 'x_restoreds'
                    for child in node.body)
        )
        block = compile(ast.Module(body=[collection]), eval_path, 'exec')
        prediction = torch.ones(1, 64, 4, 4)
        scope = {'x_restored': prediction, 'x_restoreds': []}
        exec(block, scope)
        self.assertEqual(len(scope['x_restoreds']), 1)
        self.assertEqual(scope['x_restoreds'][0].shape, (64, 4, 4))

    def test_restored_gif_skips_latent_output(self):
        draw_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'draw_eval_gifs.py')
        with open(draw_path) as draw_file:
            tree = ast.parse(draw_file.read())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'create_restored_gif')
        calls = []
        scope = {
            'draw_frames_with_restored': lambda frames, restored, *args, **kwargs: calls.append(restored),
            'os': SimpleNamespace(path=SimpleNamespace(join=lambda *parts: '/'.join(parts))),
            'imageio': SimpleNamespace(mimsave=lambda *args, **kwargs: calls.append('saved')),
        }
        exec(compile(ast.Module(body=[function]), draw_path, 'exec'), scope)
        create_restored_gif = scope['create_restored_gif']
        frames = np.zeros((1, 4, 80, 80))
        legacy_image = np.ones((1, 4, 80, 80))
        create_restored_gif(
            {'frames': frames, 'x_restoreds': np.zeros((1, 64, 4, 4))},
            'eval', 'local', stop_idx=1,
        )
        self.assertEqual(calls, [])

        create_restored_gif({'frames': frames, 'x_restoreds': legacy_image},
                            'eval', 'local', stop_idx=1)
        self.assertIs(calls[0], legacy_image)
        self.assertEqual(calls[1], 'saved')


if __name__ == "__main__":
    unittest.main()
