from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.values2 = []
        self.values_intr = []
        self.log_probs = []
        self.log_probs2 = []
        self.rewards = []
        self.entropies = []
        self.entropies2 = []
        self.actions = []
        self.actions2 = []
        self.a1_logits = []
        self.a2_logits = []
        self.a_21_logits = []
        self.betas = []
        self.states = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size
        self.action_prev = None

    def action_train(self):
        current_state = self.state.unsqueeze(0)
        model_output = self.model(
            current_state, self.hx, self.cx, None
        )

        (
            value, logit, self.hx, self.cx, _, _, value2, logit2,
            a1_logits_i, a_21_logits_i, action2, beta_active, value_intr
        ) = model_output

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, action)

        prob2 = F.softmax(logit2, dim=1)
        log_prob2 = F.log_softmax(logit2, dim=1)
        entropy2 = -(log_prob2 * prob2).sum(1)
        self.entropies2.append(entropy2)
        action2 = action2.data
        log_prob2 = log_prob2.gather(1, action2)
        self.log_probs2.append(log_prob2)
        self.values2.append(value2)
        self.values_intr.append(value_intr)
        self.actions2.append(action2)
        self.a2_logits.append(logit2)
        self.a1_logits.append(a1_logits_i)
        self.a_21_logits.append(a_21_logits_i)
        self.betas.append(beta_active)

        batch_size = 1
        num_outputs = logit.size(1)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.action_prev = torch.zeros(batch_size, num_outputs).cuda()
                self.action_prev[0, action.item()] = 1.0
        else:
            self.action_prev = torch.zeros(batch_size, num_outputs)
            self.action_prev[0, action.item()] = 1.0

        state, self.reward, self.done, self.info = self.env.step(
            action.item())
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()

        if self.done:
            self.action_prev = None
            self.model.current_option = None

        self.eps_len += 1
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.rewards.append(self.reward)
        self.states.append(current_state)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = torch.zeros(1, self.hidden_size).cuda()
                        self.hx = torch.zeros(1, self.hidden_size).cuda()
                else:
                    self.cx = torch.zeros(1, self.hidden_size)
                    self.hx = torch.zeros(1, self.hidden_size)
                self.model.current_option = None
                self.action_prev = None

            model_output = self.model(
                self.state.unsqueeze(0), self.hx, self.cx, None
            )

            value, logit, self.hx, self.cx = model_output[:4]
            prob = F.softmax(logit, dim=1)
            action = prob.cpu().numpy().argmax()

            num_outputs = logit.size(1)
            if self.gpu_id >= 0:
                with torch.cuda.device(self.gpu_id):
                    self.action_prev = torch.zeros(1, num_outputs).cuda()
                    self.action_prev[0, action] = 1.0
            else:
                self.action_prev = torch.zeros(1, num_outputs)
                self.action_prev[0, action] = 1.0
        state, self.reward, self.done, self.info = self.env.step(action)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()

        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []
        self.values2 = []
        self.values_intr = []
        self.log_probs = []
        self.log_probs2 = []
        self.rewards = []
        self.entropies = []
        self.entropies2 = []
        self.actions = []
        self.actions2 = []
        self.a1_logits = []
        self.a2_logits = []
        self.a_21_logits = []
        self.betas = []
        self.states = []
        return self
