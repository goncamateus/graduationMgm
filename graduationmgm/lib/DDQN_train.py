import os.path
import pickle

import numpy as np
import torch
import torch.optim as optim

from graduationmgm.lib.base_train import BaseTrain
from graduationmgm.lib.prioritized_experience_replay import Memory
from graduationmgm.lib.utils import MemoryDeque


class DuelingTrain(BaseTrain):
    def __init__(self, static_policy=False, env=None,
                 config=None, log_dir='/tmp/RC_test'):
        super(DuelingTrain, self).__init__(
            config=config, env=env, log_dir=log_dir)

        self.noisy = config.USE_NOISY_NETS
        self.priority_replay = config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.sigma_init = config.SIGMA_INIT
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def load_w(self, model_path=None, optim_path=None):
        if model_path is None:
            fname_model = "./saved_agents/model.dump"
        else:
            fname_model = model_path
        if optim_path is None:
            fname_optim = "./saved_agents/optim.dump"
        else:
            fname_optim = optim_path

        if os.path.isfile(fname_model):
            self.model.load_state_dict(torch.load(
                fname_model, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(
                torch.load(fname_optim, map_location=self.device))

    def declare_networks(self):
        pass

    def declare_memory(self):
        self.memory = Memory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if len(self.nstep_buffer) < self.nsteps:
            return

        R = sum([self.nstep_buffer[i][2] * (self.gamma**i)
                 for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.store((state, action, R, s_))

    def prep_minibatch(self):
        if self.priority_replay:
            # random transition batch is taken from experience replay memory
            transitions, indices, weights = self.memory.sample(self.batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)
            indices = weights = None

        batch_state = np.array([each[0][0]
                                for each in transitions], ndmin=2)
        batch_action = np.array(
            [each[0][1] for each in transitions])
        batch_reward = np.array(
            [each[0][2] for each in transitions])
        batch_next_state = np.array([each[0][3]
                                     for each in transitions], ndmin=2)
        
        with open('lalaland.pkl', 'wb') as fiile:
            pickle.dump(batch_state, fiile)

        shape = (self.batch_size,) + self.num_feats
        batch_state = torch.from_numpy(
            batch_state).float().view(shape).to(self.device)
        batch_action = torch.tensor(
            batch_action, device=self.device,
            dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(
            batch_reward, device=self.device,
            dtype=torch.float).squeeze().view(-1, 1)

        if weights is not None:
            weights = torch.tensor(weights, device=self.device)

        non_final_mask = torch.tensor(tuple(map(
            lambda s: s is not None, batch_next_state)),
            device=self.device, dtype=torch.uint8)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor(
                [s for s in batch_next_state if s is not None],
                device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except Exception:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states,\
            non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):  # faster
        batch_state, batch_action, batch_reward, non_final_next_states,\
            non_final_mask, empty_next_state_values,\
            indices, weights = batch_vars

        # estimate
        current_q_values = self.model(batch_state)
        current_q_values = current_q_values.gather(1, batch_action)

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(
                self.batch_size, device=self.device,
                dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(
                    non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(
                    non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + \
                ((self.gamma**self.nsteps) * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.batch_update(
                indices, diff.detach().squeeze().abs().cpu().numpy())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1):  # faster
        with torch.no_grad():
            if np.random.uniform() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                out = self.model(X)
                maxout = out.max(0)
                maxout = maxout[0]
                maxout = maxout.max(0)[1]
                a = maxout.view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma**i)
                     for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.store((state, action, R, None))

    def reset_hx(self):
        pass
