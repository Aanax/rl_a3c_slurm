from types import SimpleNamespace
import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import model
from oracle_loss import compute_oracle_losses


class _FutureModel(model._FutureSharedDiffTargetMixin):
    def __init__(self):
        pass


class OracleLossTest(unittest.TestCase):
    def test_two_step_reverse_recurrence_and_restoration_gradients(self):
        for future in (False, True):
            with self.subTest(future=future):
                values = [
                    torch.tensor([0.25], requires_grad=True),
                    torch.tensor([0.5], requires_grad=True),
                    torch.tensor([0.0]),
                ]
                intrinsic_values = [
                    torch.tensor([0.1], requires_grad=True),
                    torch.tensor([0.2], requires_grad=True),
                    torch.tensor([0.0]),
                ]
                log_probs = [
                    torch.tensor([0.2], requires_grad=True),
                    torch.tensor([0.3], requires_grad=True),
                ]
                restored = [
                    torch.tensor([0.0, 1.0], requires_grad=True),
                    torch.tensor([1.0, 0.0], requires_grad=True),
                ]
                player = SimpleNamespace(
                    model=_FutureModel() if future else SimpleNamespace(),
                    rewards=[torch.tensor([1.0]), torch.tensor([2.0])],
                    values=values,
                    values_intrinsic=intrinsic_values,
                    x_restoreds=restored,
                    states=[torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
                    next_states=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
                    shared_states=[
                        torch.tensor([0.0, 0.0]),
                        torch.tensor([1.0, 0.0]),
                        torch.tensor([1.0, 1.0]),
                    ],
                    log_probs=log_probs,
                    entropies=[torch.tensor([0.0]), torch.tensor([0.0])],
                )
                args = SimpleNamespace(
                    gamma=0.5,
                    gamma_restoration=0.5,
                    tau=0.5,
                    w_restoration_loss=1.0,
                    relu_g_const=False,
                    delta_t_mode='td',
                    empirical_distribution_correction=False,
                    empirical_distribution_correction_epsilon=0.0,
                    entropy_coef=0.0,
                )
                policy_loss, value_loss, intrinsic_loss, restoration_loss, cosines = compute_oracle_losses(
                    args, player, torch.tensor([0.0]), torch.tensor([0.0]), 1.0
                )
                total = policy_loss + 0.5 * value_loss + restoration_loss + 0.5 * intrinsic_loss
                total.backward()

                self.assertEqual(len(cosines), 2)
                self.assertAlmostEqual(cosines[0], 0.4472136, places=5)
                self.assertAlmostEqual(cosines[1], 0.3713907, places=5)
                self.assertAlmostEqual(total.item(), -0.474596, places=5)
                self.assertAlmostEqual(log_probs[0].grad.item(), -1.566597, places=5)
                self.assertAlmostEqual(log_probs[1].grad.item(), -1.523607, places=5)
                self.assertAlmostEqual(values[0].grad.item(), -0.875, places=5)
                self.assertAlmostEqual(values[1].grad.item(), -0.75, places=5)
                self.assertAlmostEqual(intrinsic_values[0].grad.item(), -0.098749, places=5)
                self.assertAlmostEqual(intrinsic_values[1].grad.item(), -0.011803, places=5)
                self.assertNotEqual(restored[0].grad.abs().sum().item(), 0.0)
                self.assertNotEqual(restored[1].grad.abs().sum().item(), 0.0)

    def test_fixed_losses_and_gradients(self):
        cases = (
            (False, 'td', True, 1.0, False, -0.799375, -1.15),
            (True, 'td', True, 1.0, False, -0.799375, -1.15),
            (False, 'advantage', True, 1.0, True, -0.743432, -0.870287),
            (False, 'td', False, 0.0, False, -0.009375, -0.75),
        )
        for future, delta_t_mode, restore, w_intrinsic, edc, expected_total, expected_log_grad in cases:
            with self.subTest(future=future, delta_t_mode=delta_t_mode, restore=restore, w_intrinsic=w_intrinsic, edc=edc):
                value = torch.tensor([0.25], requires_grad=True)
                intrinsic_value = torch.tensor([0.1], requires_grad=True)
                log_prob = torch.tensor([0.2], requires_grad=True)
                restored = torch.tensor([3.0, 4.0], requires_grad=True)
                player = SimpleNamespace(
                    model=_FutureModel() if future else SimpleNamespace(),
                    rewards=[torch.tensor([1.0])],
                    values=[value, torch.tensor([0.0])],
                    values_intrinsic=[intrinsic_value, torch.tensor([0.0])],
                    x_restoreds=[restored] if restore else [],
                    states=[torch.tensor([0.0, 0.0])],
                    next_states=[torch.tensor([3.0, 4.0])],
                    shared_states=[torch.tensor([0.0, 0.0]), torch.tensor([3.0, 4.0])],
                    log_probs=[log_prob],
                    entropies=[torch.tensor([0.0])],
                )
                args = SimpleNamespace(
                    gamma=0.5,
                    gamma_restoration=0.5,
                    tau=1.0,
                    w_restoration_loss=1.0 if restore else 0.0,
                    relu_g_const=False,
                    delta_t_mode=delta_t_mode,
                    empirical_distribution_correction=edc,
                    empirical_distribution_correction_epsilon=0.1,
                    entropy_coef=0.0,
                )
                losses = compute_oracle_losses(
                    args, player, torch.tensor([0.0]), torch.tensor([0.0]), w_intrinsic
                )
                policy_loss, value_loss, intrinsic_loss, restoration_loss, cosine_values = losses
                total = policy_loss + 0.5 * value_loss + restoration_loss + 0.5 * intrinsic_loss * w_intrinsic
                total.backward()

                self.assertAlmostEqual(total.item(), expected_total, places=5)
                self.assertAlmostEqual(log_prob.grad.item(), expected_log_grad, places=5)
                self.assertAlmostEqual(value.grad.item(), -0.375, places=5)
                intrinsic_grad = 0.0 if intrinsic_value.grad is None else intrinsic_value.grad.item()
                self.assertAlmostEqual(intrinsic_grad, -0.2 if restore else 0.0, places=5)
                self.assertEqual(len(cosine_values), 1 if restore else 0)


if __name__ == '__main__':
    unittest.main()
