import ast
import math
import os
import unittest
from types import SimpleNamespace

import torch


def _policy_loss_block():
    train_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py')
    with open(train_path) as train_file:
        tree = ast.parse(train_file.read())
    body = next(
        body for node in ast.walk(tree) if hasattr(node, 'body')
        for body in [node.body] if isinstance(body, list)
        if any(isinstance(statement, ast.Assign)
               and any(isinstance(target, ast.Name)
                       and target.id == 'policy_advantage'
                       for target in statement.targets)
               for statement in body))
    index = next(index for index, statement in enumerate(body)
                 if isinstance(statement, ast.Assign)
                 and any(isinstance(target, ast.Name)
                         and target.id == 'policy_advantage'
                         for target in statement.targets))
    module = ast.Module(body=body[index:index + 3])
    return compile(module, train_path, 'exec')


class EmpiricalDistributionCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.block = _policy_loss_block()

    def _loss_and_gradient(self, correction, epsilon, probability, advantage=2.0,
                           logprob_value=None):
        logprob = torch.tensor(math.log(probability) if logprob_value is None
                               else logprob_value, dtype=torch.float64,
                               requires_grad=True)
        namespace = {
            'args': SimpleNamespace(
                empirical_distribution_correction=correction,
                empirical_distribution_correction_epsilon=epsilon,
                entropy_coef=0.0),
            'player': SimpleNamespace(log_probs=[logprob],
                                      entropies=[torch.tensor(0.0, dtype=torch.float64)]),
            'i': 0,
            'gae': torch.tensor(advantage, dtype=torch.float64),
            'gae_intrinsic': torch.tensor(0.0, dtype=torch.float64),
            'policy_loss': torch.tensor(0.0, dtype=torch.float64),
        }
        exec(self.block, namespace)
        loss = namespace['policy_loss']
        return loss.item(), torch.autograd.grad(loss, logprob)[0].item()

    def test_epsilon_correction_uses_the_extracted_training_block(self):
        probability = 0.25
        advantage = 2.0
        original_loss = -math.log(probability) * advantage

        for correction, epsilon in ((False, 0.001), (True, 0.0)):
            loss, gradient = self._loss_and_gradient(correction, epsilon,
                                                     probability, advantage)
            expected_weight = 1.0 / probability if correction else 1.0
            self.assertAlmostEqual(loss, original_loss * expected_weight, places=12)
            self.assertAlmostEqual(gradient, -advantage * expected_weight, places=12)

        epsilon = 0.001
        loss, gradient = self._loss_and_gradient(True, epsilon, probability,
                                                 advantage)
        weight = 1.0 / (probability + epsilon)
        self.assertAlmostEqual(loss, original_loss * weight, places=12)
        self.assertAlmostEqual(gradient, -advantage * weight, places=12)

    def test_positive_epsilon_bounds_tiny_probability_weight(self):
        loss, gradient = self._loss_and_gradient(True, 0.001, 1.0, 1.0,
                                                 logprob_value=-1000.0)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(math.isfinite(gradient))
        self.assertLessEqual(abs(gradient), 1000.0)


if __name__ == '__main__':
    unittest.main()
