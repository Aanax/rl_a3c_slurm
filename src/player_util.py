from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F

from model import HierarchialLevelsOutput


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
        self.values_int = []
        self.log_probs = []
        self.log_probs2 = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions2 = []
        self.a1_logits = []
        self.a2_logits = []
        self.option_terminated = []
        self.action_terminated = []
        self.betas = []
        self.betas2 = []
        self.states = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size
        self.action_prev = None

    def _reset_persistent_actions(self):
        if hasattr(self.model, 'reset_persistent_actions'):
            self.model.reset_persistent_actions()
        elif hasattr(self.model, 'current_option'):
            self.model.current_option = None

    def action_train(self):
        current_state = self.state.unsqueeze(0)
        model_output = self.model(
            current_state, self.hx, self.cx, None
        )

        if isinstance(model_output, HierarchialLevelsOutput):
            value = model_output.V1
            value_int = model_output.V1_int
            logit = model_output.a1_logits
            self.hx = model_output.hx
            self.cx = model_output.cx
            value2 = model_output.V2
            logit2 = model_output.a2_logits
            action = model_output.a1.data
            action2 = model_output.a2
            self.betas2.append(model_output.beta2)
            self.betas.append(model_output.beta2)
            self.action_terminated.append(model_output.terminated1)
            option_terminated = model_output.terminated2

            log_prob_all = F.log_softmax(logit, dim=1)
            prob = F.softmax(logit, dim=1)
            log_prob = log_prob_all.gather(1, action)
            entropy = -log_prob
        else:
            value_int = None
            (
                value, logit, self.hx, self.cx, _, _, value2, logit2,
                action2, beta_active, option_terminated
            ) = model_output
            self.betas.append(beta_active)
            self.betas2.append(beta_active)
            self.action_terminated.append(False)

            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, action)
            entropy = -log_prob

        self.entropies.append(entropy)

        log_prob2 = F.log_softmax(logit2, dim=1)
        action2 = action2.data
        log_prob2 = log_prob2.gather(1, action2)
        self.log_probs2.append(log_prob2)
        self.values2.append(value2)
        self.actions2.append(action2)
        self.a2_logits.append(logit2)
        self.option_terminated.append(option_terminated)

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
            self._reset_persistent_actions()

        self.eps_len += 1
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        if value_int is not None:
            self.values_int.append(value_int)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.a1_logits.append(logit)
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
                self._reset_persistent_actions()
                self.action_prev = None

            model_output = self.model(
                self.state.unsqueeze(0), self.hx, self.cx, None
            )

            if isinstance(model_output, HierarchialLevelsOutput):
                logit = model_output.a1_logits
                self.hx = model_output.hx
                self.cx = model_output.cx
                action = model_output.a1.item()
            else:
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
        self.values_int = []
        self.log_probs = []
        self.log_probs2 = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions2 = []
        self.a1_logits = []
        self.a2_logits = []
        self.option_terminated = []
        self.action_terminated = []
        self.betas = []
        self.betas2 = []
        self.states = []
        return self
