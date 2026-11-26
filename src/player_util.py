"""
The agent: plays the game with the model and writes down everything train.py needs.

action_train() plays one step while training and stores it.
action_test()  plays one step greedily (used by test.py), stores nothing.
clear_actions() empties the stored steps after each training batch.
"""
from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.args = args
        self.state = state
        self.done = True
        self.info = None
        self.reward = 0
        self.eps_len = 0
        # Memory: s1 of the previous step. None at the start of an episode.
        self.previous_s1 = None
        # s1 of every step of the current game, for monitoring. train.py saves and clears it.
        self.s1_history = []
        self.clear_actions()

    def action_train(self):
        state = self.state.unsqueeze(0)
        out = self.model(state, self.previous_s1)

        # Level 1: sample the game action
        action_prob = F.softmax(out['action_logits'], dim=1)
        action_log_prob = F.log_softmax(out['action_logits'], dim=1)
        action_entropy = -(action_log_prob * action_prob).sum(1)
        action = action_prob.multinomial(1).data

        # Level 2: the option was already sampled inside the model
        option = out['option']
        option_prob = F.softmax(out['option_logits'], dim=1)
        option_log_prob = F.log_softmax(out['option_logits'], dim=1)
        option_entropy = -(option_log_prob * option_prob).sum(1)

        # If the option has just changed, the previous option ended at this state.
        # Ask the model what the previous option would see here: these values close
        # the previous option's returns in train.py. They go into the previous step's slot.
        if len(self.options) > 0 and option.item() != self.options[-1].item():
            with torch.no_grad():
                finished = self.model(state, self.previous_s1, option=self.options[-1])
            self.option_end_values[-1] = finished['value']
            self.option_end_values_intrinsic[-1] = finished['value_intrinsic']
            self.option_end_oracle1[-1] = finished['oracle1_prediction']

        # Play the action
        next_state, self.reward, self.done, self.info = self.env.step(action.item())
        self.state = torch.from_numpy(next_state).float()
        self.reward = max(min(self.reward, 1), -1)
        self.eps_len += 1

        # Update the memory. It starts again from zero after a lost life.
        if self.done:
            self.previous_s1 = None
        else:
            self.previous_s1 = out['s1']

        # Store this step
        self.rewards.append(self.reward)
        self.log_probs.append(action_log_prob.gather(1, action))
        self.entropies.append(action_entropy)
        self.values.append(out['value'])
        self.values_intrinsic.append(out['value_intrinsic'])
        self.oracle1_predictions.append(out['oracle1_prediction'])
        self.s1_states.append(out['s1'].detach())
        self.s1_history.append(out['s1'].detach())

        self.options.append(option)
        self.log_probs2.append(option_log_prob.gather(1, option))
        self.entropies2.append(option_entropy)
        self.values2.append(out['value2'])
        self.values_intrinsic2.append(out['value_intrinsic2'])
        self.oracle2_predictions.append(out['oracle2_prediction'])
        self.s2_states.append(out['s2'])

        # Filled in later, only if the option changes on the next step
        self.option_end_values.append(None)
        self.option_end_values_intrinsic.append(None)
        self.option_end_oracle1.append(None)

        # After a lost life there is no next state: its features count as zero
        if self.done:
            self.s1_states.append(torch.zeros_like(out['s1']))
            self.s2_states.append(torch.zeros_like(out['s2']))
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                self.previous_s1 = None
            out = self.model(self.state.unsqueeze(0), self.previous_s1)
            self.previous_s1 = out['s1']
            action = out['action_logits'].argmax(dim=1).item()
        next_state, self.reward, self.done, self.info = self.env.step(action)
        self.state = torch.from_numpy(next_state).float()
        self.eps_len += 1
        return self

    def clear_actions(self):
        # Level 1, one entry per step
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        self.values = []
        self.values_intrinsic = []
        self.oracle1_predictions = []
        # Level 2, one entry per step
        self.options = []
        self.log_probs2 = []
        self.entropies2 = []
        self.values2 = []
        self.values_intrinsic2 = []
        self.oracle2_predictions = []
        # Features, one entry per step plus one for the state after the last step
        self.s1_states = []
        self.s2_states = []
        # Values of a finished option at the state where it ended
        self.option_end_values = []
        self.option_end_values_intrinsic = []
        self.option_end_oracle1 = []
        return self
