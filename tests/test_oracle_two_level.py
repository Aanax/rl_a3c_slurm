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
        "level2_batch_return_step",
        "level1_option_delta",
        "option_truncated_gae",
        "oracle_option_target",
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
level2_batch_return_step = _HELPERS["level2_batch_return_step"]
level1_option_delta = _HELPERS["level1_option_delta"]
option_truncated_gae = _HELPERS["option_truncated_gae"]
oracle_option_target = _HELPERS["oracle_option_target"]
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
        self.assertEqual(net.level2_encoder.conv1.in_channels, 64)
        self.assertEqual(net.level2_encoder.conv1.out_channels, 32)
        self.assertEqual(net.actor_linear2.in_features, 32 * 4 * 4)
        self.assertEqual(net.critic_linear2.in_features, 32 * 4 * 4)
        self.assertEqual(net.actor_linear.in_features, 64 * 4 * 4 + num_options)
        self.assertEqual(net.critic_linear.in_features, 64 * 4 * 4 + num_options)
        self.assertEqual(
            net.critic_linear_intrinsic.in_features,
            64 * 4 * 4 + num_options,
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

        self.assertEqual(int(option0[8].item()), 0)
        self.assertEqual(int(option2[8].item()), 2)
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

    def test_level1_td_delta_drops_the_next_option_value(self):
        ended = level1_option_delta(
            torch.tensor(1.0), 0.5, torch.tensor(0.25), torch.tensor(8.0), True,
        )
        continued = level1_option_delta(
            torch.tensor(1.0), 0.5, torch.tensor(0.25), torch.tensor(8.0), False,
        )
        self.assertAlmostEqual(float(ended.item()), 1.0 - 0.25)
        self.assertAlmostEqual(float(continued.item()), 1.0 + 0.5 * 8.0 - 0.25)

    def test_gae_drops_advantage_carried_from_the_next_option(self):
        ended = option_truncated_gae(
            torch.tensor(5.0), torch.tensor(1.0), 0.5, 1.0, True,
        )
        continued = option_truncated_gae(
            torch.tensor(5.0), torch.tensor(1.0), 0.5, 1.0, False,
        )
        self.assertAlmostEqual(float(ended.item()), 1.0)
        self.assertAlmostEqual(float(continued.item()), 0.5 * 5.0 + 1.0)

    def test_bootstrap_option_index_conditions_level1_critics(self):
        num_options = 4
        net = model.A3CRules2378OracleTwoLevel(1, SimpleNamespace(n=3), _args(num_options))
        net.train()
        net.actor_linear2.weight.data.zero_()
        net.actor_linear2.bias.data.zero_()
        net.actor_linear2.bias.data[0] = 10.0
        for head in (net.critic_linear, net.critic_linear_intrinsic):
            head.weight.data.zero_()
            head.bias.data.zero_()
            head.weight.data[0, -num_options:] = torch.arange(
                num_options, dtype=head.weight.dtype,
            )
        observation = torch.zeros(1, 1, 80, 80)

        output = net(
            observation, None, None, option_index=torch.tensor([[2]]),
        )

        self.assertEqual(int(output[8].item()), 2)
        self.assertAlmostEqual(float(output[0].item()), 2.0)
        self.assertAlmostEqual(float(output[5].item()), 2.0)

    def test_level2_return_crosses_option_boundaries(self):
        gamma = 0.5
        rewards = [torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])]
        running = torch.tensor([[10.0]])
        for reward in reversed(rewards):
            running = level2_batch_return_step(running, reward, gamma)
        self.assertAlmostEqual(
            float(running.item()),
            1.0 + gamma * 2.0 + gamma ** 2 * 3.0 + gamma ** 3 * 10.0,
        )

    def test_train_level2_return_runs_through_the_batch(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
        self.assertIn("R2 = level2_batch_return_step(R2, r2_i, args.gamma2)", source)

    def test_oracle_target_is_zero_at_option_end(self):
        gamma = 0.5
        deltas = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
        option_ids = [0, 0, 1]
        running = torch.tensor([10.0])
        targets = [None] * len(deltas)
        for index in reversed(range(len(deltas))):
            running = oracle_option_target(
                running,
                deltas[index],
                gamma,
                option_changes_after(option_ids, index),
            )
            targets[index] = running.clone()

        self.assertTrue(torch.equal(targets[2], torch.tensor([gamma * 10.0 + 3.0])))
        self.assertTrue(torch.equal(targets[1], torch.zeros(1)))
        self.assertTrue(torch.equal(targets[0], deltas[0]))

    def test_oracle_target_keeps_a_vector_shape_when_the_option_ends(self):
        ended = oracle_option_target(
            torch.ones(1, 2, 2), torch.full((1, 2, 2), 3.0), 0.75, True,
        )
        self.assertTrue(torch.equal(ended, torch.zeros(1, 2, 2)))
        self.assertFalse(ended.requires_grad)

    def test_train_cuts_oracle_target_at_option_end(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
        self.assertIn("oracle_bootstrap = model_output[4].detach()", source)
        self.assertIn("oracle_option_target(", source)
        self.assertIn("player.done and i + 1 == len(player.rewards)", source)

    def test_hidden_state_oracles_match_s1_and_s2(self):
        net = model.A3CRules2378OracleTwoLevel(
            1, SimpleNamespace(n=3), _args(4),
        )
        self.assertTrue(net.predicts_shared_diff)
        self.assertFalse(any(
            name.startswith("decoder") for name, _parameter in net.named_parameters()
        ))
        observation = torch.ones(1, 1, 80, 80)
        first = net(observation, None, None, option_index=0)
        second = net(observation, None, None, option_index=3)
        self.assertEqual(tuple(first[4].shape), (1, 64, 4, 4))
        self.assertEqual(tuple(first[-3].shape), (1, 64, 4, 4))
        self.assertEqual(tuple(first[-2].shape), tuple(first[-1].shape))
        self.assertEqual(tuple(first[-1].shape), (1, 32, 4, 4))
        self.assertEqual(net.oracle2_head.in_channels, 32)
        self.assertEqual(int(first[8].item()), 0)
        self.assertEqual(int(second[8].item()), 3)
        self.assertEqual(tuple(first[-4].shape), (1, 1))
        self.assertTrue(torch.equal(first[-4], second[-4]))
        self.assertEqual(
            net.critic_linear_intrinsic2.in_features,
            net.critic_linear2.in_features,
        )
        self.assertTrue(torch.equal(first[4], second[4]))
        self.assertTrue(torch.equal(first[-1], second[-1]))
        self.assertFalse(first[-3].requires_grad)
        self.assertFalse(first[-2].requires_grad)
        self.assertTrue(first[4].requires_grad)
        self.assertTrue(first[-1].requires_grad)

        concat_args = _args(4)
        concat_args.actor_input_mode = "concat"
        concat_args.critic_input_mode = "concat"
        concat_net = model.A3CRules2378OracleTwoLevel(
            1, SimpleNamespace(n=3), concat_args,
        )
        concat_out = concat_net(observation, None, None, option_index=1)
        self.assertEqual(tuple(concat_out[4].shape), (1, 64, 4, 4))
        self.assertEqual(tuple(concat_out[-1].shape), (1, 32, 4, 4))
        self.assertEqual(tuple(concat_out[-2].shape), (1, 32, 4, 4))

    def test_oracle_gradients_reach_the_shared_encoder(self):
        net = model.A3CRules2378OracleTwoLevel(
            1, SimpleNamespace(n=3), _args(4),
        )
        for module in net.shared_encoder.modules():
            if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                module.bias.data.fill_(0.1)
        observation = torch.ones(1, 1, 80, 80)
        pred_s1 = net(observation, None, None, option_index=0)[4]
        pred_s1.sum().backward()
        self.assertIsNotNone(net.shared_encoder.conv1.weight.grad)
        self.assertTrue(torch.any(net.shared_encoder.conv1.weight.grad != 0))
        net.zero_grad()
        if net.prev_shared is not None:
            net.prev_shared = net.prev_shared.detach()
        net.level2_encoder.conv1.bias.data.fill_(0.1)
        pred_s2 = net(observation, None, None, option_index=0)[-1]
        pred_s2.sum().backward()
        self.assertIsNotNone(net.shared_encoder.conv1.weight.grad)
        self.assertTrue(torch.any(net.shared_encoder.conv1.weight.grad != 0))

    def test_train_level2_actor_uses_both_advantages(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
        self.assertIn("(advantage2 + advantage_intrinsic2).detach()", source)
        self.assertIn("R_intrinsic2, oracle_r2, args.gamma2", source)
        self.assertIn("(1.0 - args.gamma2) * cosine_const2", source)

    def test_train_oracle_targets_use_hidden_states(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
        self.assertIn("player.s1_states[i + 1] - player.s1_states[i]", source)
        self.assertIn("player.s2_states[i + 1] - player.s2_states[i]", source)
        self.assertIn(
            "cosine2, advantage2.detach(), args.w_restoration_loss",
            source,
        )

    def test_train_cuts_level1_critic_at_option_end(self):
        source = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
        self.assertIn("R, player.rewards[i], args.gamma, option_ended", source)
        self.assertIn("R_intrinsic, oracle_r, args.gamma, option_ended", source)
        self.assertIn("option_index=player.actions2[-1]", source)
        self.assertIn("level1_option_delta(", source)
        self.assertIn("option_truncated_gae(", source)


if __name__ == "__main__":
    unittest.main()
