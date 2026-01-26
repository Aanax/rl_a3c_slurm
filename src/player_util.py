from __future__ import division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.x_restoreds = []
        self.kls = []
        self.states = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size

    def action_train(self):

        current_state = self.state.unsqueeze(0)
        model_output = self.model(
            current_state, self.hx, self.cx, None
        )
        if len(model_output) == 4:
            value, logit, self.hx, self.cx = model_output
            x_restored = None
            kl = None
        elif len(model_output) == 5:
            value, logit, self.hx, self.cx, x_restored = model_output
            kl = None
        elif len(model_output) == 6:
            value, logit, self.hx, self.cx, x_restored, kl = model_output
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, action)
        state, self.reward, self.done, self.info = self.env.step(
            action.item())
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = torch.from_numpy(state).float().cuda()
        else:
            self.state = torch.from_numpy(state).float()

        if self.done:
            if hasattr(self.model, 'running_mem'):
                self.model.running_mem = torch.zeros_like(self.model.running_mem)
            if hasattr(self.model, 'prev_x_conv'):
                self.model.prev_x_conv = None

        self.eps_len += 1
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward)
        self.states.append(current_state)
        if x_restored is not None:
            self.x_restoreds.append(x_restored)
        if kl is not None:
            self.kls.append(kl)
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
                # Reset model internal memory if applicable
                try:
                    if hasattr(self.model, 'running_mem'):
                        self.model.running_mem = torch.zeros_like(self.model.running_mem)
                    if hasattr(self.model, 'prev_x_conv'):
                        self.model.prev_x_conv = None
                except:
                    pass
                
            model_output = self.model(
                self.state.unsqueeze(0), self.hx, self.cx, None
            )
            if len(model_output) >= 4:
                value, logit, self.hx, self.cx = model_output[:4]
            prob = F.softmax(logit, dim=1)
            action = prob.cpu().numpy().argmax()
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
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.x_restoreds = []
        self.kls = []
        self.states = []
        return self
