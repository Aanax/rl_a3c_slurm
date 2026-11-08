import ast
import os
import sys
import unittest
from types import SimpleNamespace

import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import model


class _ActionSpace:
    n = 3


def _bootstrap_block():
    train_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py')
    with open(train_path) as train_file:
        tree = ast.parse(train_file.read())
    bootstrap = next(
        node for node in ast.walk(tree) if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Attribute)
        and node.test.operand.attr == 'done'
    )
    return compile(ast.Module(body=[bootstrap]), train_path, 'exec')


class BootstrapTemporalStateTest(unittest.TestCase):
    def _args(self, mode):
        return SimpleNamespace(
            monitor_s=False,
            use_rmsnorm=False,
            actor_input_mode=mode,
            critic_input_mode='shared',
            gamma_memory=0.99,
        )

    def _model(self, cls, mode):
        torch.manual_seed(7)
        return cls(4, _ActionSpace(), self._args(mode))

    @staticmethod
    def _snapshot(net):
        return (
            net.prev_shared, net.memdiff_sum, net.memdiff_count,
            getattr(net, 'memdiff_w', None),
            getattr(net, 'memdiff_w_prev', None),
        )

    def _run_bootstrap(self, net, state):
        player = SimpleNamespace(
            done=False,
            state=state.squeeze(0),
            hx=None,
            cx=None,
            model=net,
            shared_states=[],
        )
        exec(_bootstrap_block(), {'model': model, 'player': player, 'torch': torch})

    def test_bootstrap_restores_shared_diff_state(self):
        state0 = torch.randn(1, 4, 80, 80)
        state1 = torch.randn(1, 4, 80, 80)
        classes = (
            model.A3CRules2378OracleNoSplitSharedDiff,
            model.A3CRules2378OracleNoSplitSharedFutureDiff,
        )
        for cls in classes:
            for mode in ('concat', 'concat_memdiff', 'concat_memdiff_raw'):
                with self.subTest(cls=cls.__name__, mode=mode):
                    reference = self._model(cls, mode)
                    trial = self._model(cls, mode)
                    trial.load_state_dict(reference.state_dict())
                    reference(state0, None, None)
                    reference(state1, None, None)
                    expected = reference(state1, None, None)

                    trial(state0, None, None)
                    trial(state1, None, None)
                    snapshot = self._snapshot(trial)
                    self._run_bootstrap(trial, state1)
                    self.assertIs(trial.prev_shared, snapshot[0])
                    self.assertIs(trial.memdiff_sum, snapshot[1])
                    self.assertEqual(trial.memdiff_count, snapshot[2])
                    self.assertEqual(getattr(trial, 'memdiff_w', None), snapshot[3])
                    self.assertEqual(getattr(trial, 'memdiff_w_prev', None), snapshot[4])

                    actual = trial(state1, None, None)
                    for index in (0, 1, 4):
                        self.assertTrue(torch.equal(actual[index], expected[index]))

    def test_bootstrap_restores_state_after_forward_error(self):
        net = self._model(model.A3CRules2378OracleNoSplitSharedFutureDiff, 'concat_memdiff')
        state0 = torch.randn(1, 4, 80, 80)
        state1 = torch.randn(1, 4, 80, 80)
        net(state0, None, None)
        net(state1, None, None)
        snapshot = self._snapshot(net)

        def fail(actor_feat):
            raise RuntimeError('oracle failure')

        net._oracle_output = fail
        with self.assertRaises(RuntimeError):
            self._run_bootstrap(net, state1)
        self.assertIs(net.prev_shared, snapshot[0])
        self.assertIs(net.memdiff_sum, snapshot[1])
        self.assertEqual(net.memdiff_count, snapshot[2])
        self.assertEqual(getattr(net, 'memdiff_w', None), snapshot[3])
        self.assertEqual(getattr(net, 'memdiff_w_prev', None), snapshot[4])


if __name__ == '__main__':
    unittest.main()
