import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model


def _load_train_helpers():
    train_path = Path(__file__).resolve().parents[1] / "src" / "train.py"
    tree = ast.parse(train_path.read_text())
    wanted = {
        "_option_index",
        "option_changes_after",
        "option_segment_return_step",
        "oracle_two_level_actor_loss",
        "external_advantage_restoration",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(train_path), "exec"), namespace)
    return namespace


_HELPERS = _load_train_helpers()
option_changes_after = _HELPERS["option_changes_after"]
option_segment_return_step = _HELPERS["option_segment_return_step"]
oracle_two_level_actor_loss = _HELPERS["oracle_two_level_actor_loss"]
external_advantage_restoration = _HELPERS["external_advantage_restoration"]


def _args(num_options):
    return SimpleNamespace(
        monitor_s=False,
        use_rmsnorm=False,
        num_options=num_options,
        actor_input_mode="shared",
        critic_input_mode="shared",
        gamma_memory=0.0,
        hidden_size=32,
    )


def _walk_option_returns(rewards, option_ids, gamma, bootstrap):
    running = bootstrap.clone()
    returns = [None] * len(rewards)
    for index in reversed(range(len(rewards))):
        running = option_segment_return_step(
            running,
            rewards[index],
            gamma,
            option_changes_after(option_ids, index),
        )
        returns[index] = running.clone()
    return returns


class OracleTwoLevelTest(unittest.TestCase):
    def test_option_onehot_conditions_level1_heads_and_not_the_decoder(self):
        num_options = 4
        net = model.A3CRules2378OracleTwoLevel(1, SimpleNamespace(n=3), _args(num_options))
        parameter_names = [name for name, _parameter in net.named_parameters()]
        self.assertFalse(any("beta" in name for name in parameter_names))
        self.assertEqual(
            net.actor_linear.in_features,
            net.actor_linear2.in_features + num_options,
        )
        self.assertEqual(
            net.critic_linear.in_features,
            net.critic_linear2.in_features + num_options,
        )
        self.assertEqual(
            net.critic_linear_intrinsic.in_features,
            net.actor_linear2.in_features + num_options,
        )

        net.eval()
        net.actor_linear2.weight.data.zero_()
        net.actor_linear2.bias.data.zero_()
        net.critic_linear.weight.data.zero_()
        net.critic_linear.bias.data.zero_()
        net.critic_linear.weight.data[0, -num_options:] = torch.arange(
            num_options, dtype=net.critic_linear.weight.dtype
        )
        observation = torch.zeros(1, 1, 80, 80)

        def forward_option(option_index):
            net.actor_linear2.bias.data.zero_()
            net.actor_linear2.bias.data[option_index] = 10.0
            return net(observation, None, None)

        with torch.no_grad():
            option0 = forward_option(0)
            option2 = forward_option(2)

        self.assertEqual(int(option0[-1].item()), 0)
        self.assertEqual(int(option2[-1].item()), 2)
        self.assertAlmostEqual(float(option0[0].item()), 0.0)
        self.assertAlmostEqual(float(option2[0].item()), 2.0)
        self.assertTrue(torch.equal(option0[4], option2[4]))

    def test_option_return_covers_only_that_option(self):
        gamma = 0.5
        rewards = [torch.tensor([[float(value)]]) for value in (1.0, 2.0, 3.0, 4.0, 5.0)]
        option_ids = [0, 0, 1, 1, 1]
        returns = _walk_option_returns(
            rewards, option_ids, gamma, torch.tensor([[10.0]])
        )

        self.assertAlmostEqual(float(returns[1].item()), 2.0)
        self.assertAlmostEqual(float(returns[0].item()), 1.0 + gamma * 2.0)
        option1_start = (
            3.0 + gamma * 4.0 + gamma ** 2 * 5.0 + gamma ** 3 * 10.0
        )
        self.assertAlmostEqual(float(returns[2].item()), option1_start)
        self.assertNotAlmostEqual(float(returns[0].item()), option1_start)

    def test_two_step_and_three_step_options_do_not_share_rewards(self):
        gamma = 0.25
        two_step = _walk_option_returns(
            [torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[9.0]])],
            [4, 4, 7],
            gamma,
            torch.tensor([[0.0]]),
        )
        self.assertAlmostEqual(float(two_step[1].item()), 2.0)
        self.assertAlmostEqual(float(two_step[0].item()), 1.0 + gamma * 2.0)

        three_step = _walk_option_returns(
            [torch.tensor([[3.0]]), torch.tensor([[4.0]]), torch.tensor([[5.0]])],
            [1, 1, 1],
            gamma,
            torch.tensor([[8.0]]),
        )
        self.assertAlmostEqual(float(three_step[2].item()), 5.0 + gamma * 8.0)
        self.assertAlmostEqual(
            float(three_step[0].item()),
            3.0 + gamma * 4.0 + gamma ** 2 * 5.0 + gamma ** 3 * 8.0,
        )

    def test_actor_losses_use_internal_and_option_advantages(self):
        log_prob1 = torch.tensor([[-0.2]], requires_grad=True)
        log_prob2 = torch.tensor([[-0.4]], requires_grad=True)
        advantage_int = torch.tensor([[2.0]])
        advantage2 = torch.tensor([[3.0]])
        entropy1 = torch.tensor([0.5])
        entropy2 = torch.tensor([0.25])
        policy1, policy2 = oracle_two_level_actor_loss(
            log_prob1, advantage_int, entropy1,
            log_prob2, advantage2, entropy2, entropy_coef=0.1,
        )
        self.assertTrue(torch.allclose(
            policy1, -(log_prob1 * advantage_int) - 0.1 * entropy1
        ))
        self.assertTrue(torch.allclose(
            policy2, -(log_prob2 * advantage2) - 0.1 * entropy2
        ))
        policy1.sum().backward()
        self.assertTrue(torch.allclose(log_prob1.grad, -advantage_int))

    def test_restoration_is_scaled_by_the_external_advantage(self):
        cosine_loss = torch.tensor(-0.25)
        advantage_ext = torch.tensor(4.0)
        term = external_advantage_restoration(cosine_loss, advantage_ext, 2.0)
        self.assertAlmostEqual(float(term.item()), -2.0)


if __name__ == "__main__":
    unittest.main()
