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
        self.values2 = []  # V2 values for hierarchical models
        self.log_probs = []
        self.log_probs2 = []  # Actor2 log probs for hierarchical models
        self.rewards = []
        self.entropies = []
        self.entropies2 = []  # Actor2 entropies for hierarchical models
        self.actions = []
        self.actions2 = []
        self.a1_logits = []
        self.a2_logits = []
        self.a_21_logits = []
        self.x_restoreds = []
        self.kls = []
        self.states = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.hidden_size = args.hidden_size
        self.action_prev = None  # Previous level 1 action logits (a1) as one-hot vector of size num_outputs

    def action_train(self):
        # Prepare action_prev - convert from one-hot vector to pass to model
        # Only pass action_prev for models with action memory (Hierarchial_memory_action_memrelu)
        action_prev = None
        if hasattr(self.args, 'model_type') and self.args.model_type == 'Hierarchial_memory_action_memrelu':
            action_prev = self.action_prev

            current_state = self.state.unsqueeze(0)
            model_output = self.model(
                current_state, self.hx, self.cx, None, action_prev
            )
        else:
            current_state = self.state.unsqueeze(0)
            model_output = self.model(
                    current_state, self.hx, self.cx, None
                )

        a1_logits_i = None
        a_21_logits_i = None
        action2 = None
        if len(model_output) == 4:
            value, logit, self.hx, self.cx = model_output
            x_restored = None
            kl = None
            value2 = None
            logit2 = None
        elif len(model_output) == 5:
            value, logit, self.hx, self.cx, x_restored = model_output
            kl = None
            value2 = None
            logit2 = None
        elif len(model_output) == 6:
            value, logit, self.hx, self.cx, x_restored, kl = model_output
            value2 = None
            logit2 = None
        elif len(model_output) == 8:
            value, logit, self.hx, self.cx, _, _, value2, logit2 = model_output
            x_restored = None
            kl = None
        elif len(model_output) == 10:
            value, logit, self.hx, self.cx, _, _, value2, logit2, a1_logits_i, a_21_logits_i = model_output
            x_restored = None
            kl = None
            action2 = None
        elif len(model_output) == 11:
            value, logit, self.hx, self.cx, _, _, value2, logit2, a1_logits_i, a_21_logits_i, action2 = model_output
            x_restored = None
            kl = None
        else:
            raise ValueError(f"Unexpected model output length: {len(model_output)}. Expected 4, 5, 6, 8, 10, or 11.")
        
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, action)
        
        # Handle actor2 if hierarchical model
        if logit2 is not None:
            prob2 = F.softmax(logit2, dim=1)
            log_prob2 = F.log_softmax(logit2, dim=1)
            entropy2 = -(log_prob2 * prob2).sum(1)
            self.entropies2.append(entropy2)
            if action2 is None:
                action2 = prob2.multinomial(1).data
            else:
                action2 = action2.data
            log_prob2 = log_prob2.gather(1, action2)
            self.log_probs2.append(log_prob2)
            self.values2.append(value2)
            self.actions2.append(action2)
            self.a2_logits.append(logit2)

        if a1_logits_i is not None:
            self.a1_logits.append(a1_logits_i)
        if a_21_logits_i is not None:
            self.a_21_logits.append(a_21_logits_i)

        # chosen action to onehot
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
            # Reset action_prev when episode ends
            self.action_prev = None
            if hasattr(self.model, 'running_mem'):
                self.model.running_mem = torch.zeros_like(self.model.running_mem)
            if hasattr(self.model, 'prev_x_conv'):
                self.model.prev_x_conv = None

        self.eps_len += 1
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.actions.append(action)
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
                # Reset action_prev when episode ends
                self.action_prev = None
                
            # Prepare action_prev for model
            # Only pass action_prev for models with action memory (Hierarchial_memory_action_memrelu)
            action_prev = None
            if hasattr(self.args, 'model_type') and self.args.model_type == 'Hierarchial_memory_action_memrelu':
                action_prev = self.action_prev
            
                model_output = self.model(
                    self.state.unsqueeze(0), self.hx, self.cx, None, action_prev
                )
            else:
                model_output = self.model(
                self.state.unsqueeze(0), self.hx, self.cx, None
            )

            if len(model_output) >= 4:
                # For hierarchical models (8 elements), use first 4: V1, a1, hx, cx
                # For non-hierarchical models (4-6 elements), use all available
                value, logit, self.hx, self.cx = model_output[:4]
            prob = F.softmax(logit, dim=1)
            action = prob.cpu().numpy().argmax()
            
            # Store action_prev as one-hot vector for next timestep
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
        self.x_restoreds = []
        self.kls = []
        self.states = []
        return self