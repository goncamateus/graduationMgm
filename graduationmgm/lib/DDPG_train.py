import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from graduationmgm.lib.base_train import BaseTrain
from graduationmgm.lib.prioritized_experience_replay import Memory
from graduationmgm.lib.utils import MemoryDeque


class DDPGTrain(BaseTrain):
    def __init__(self, static_policy=False, env=None,
                 config=None, log_dir='/tmp/RC_test'):
        super(DDPGTrain, self).__init__(
            config=config, env=env, log_dir=log_dir)

        self.priority_replay = config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA
        self.tau = config.tau
        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()
        actor_learning_rate = 1e-4
        critic_learning_rate = 1e-3

        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate)

        # move to correct device
        self.actor = self.actor.to(self.device)
        self.target_actor = self.target_actor.to(self.device)
        self.critic = self.critic.to(self.device)
        self.target_critic = self.target_critic.to(self.device)

        if self.static_policy:
            self.actor.eval()
            self.target_actor.eval()
            self.critic.eval()
            self.target_critic.eval()
        else:
            self.actor.train()
            self.target_actor.train()
            self.critic.train()
            self.target_critic.train()

        self.update_count = 0

        self.declare_memory()

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def save_w(self, path_models=('./saved_agents/actor.dump',
                                  './saved_agents/critic.dump'),
               path_optims=('./saved_agents/actor_optim.dump',
                            './saved_agents/critic_optim.dump')):
        torch.save(self.actor.state_dict(), path_models[0])
        torch.save(self.critic.state_dict(), path_models[1])
        torch.save(self.actor_optimizer.state_dict(), path_optims[0])
        torch.save(self.critic_optimizer.state_dict(), path_optims[1])

    def load_w(self, path_models=('./saved_agents/actor.dump',
                                  './saved_agents/critic.dump'),
               path_optims=('./saved_agents/actor_optim.dump',
                            './saved_agents/critic_optim.dump')):
        fname_actor = path_models[0]
        fname_critic = path_models[1]
        fname_actor_optim = path_optims[0]
        fname_critic_optim = path_optims[1]

        if os.path.isfile(fname_actor):
            self.actor.load_state_dict(torch.load(fname_actor,
                                                  map_location=self.device))
            for target_param, param in zip(self.target_actor.parameters(),
                                           self.actor.parameters()):
                target_param.data.copy_(param.data)

        if os.path.isfile(fname_critic):
            self.critic.load_state_dict(torch.load(fname_critic,
                                                   map_location=self.device))
            for target_param, param in zip(self.target_critic.parameters(),
                                           self.critic.parameters()):
                target_param.data.copy_(param.data)

        if os.path.isfile(fname_actor_optim):
            self.actor_optimizer.load_state_dict(
                torch.load(fname_actor_optim,
                           map_location=self.device)
            )

        if os.path.isfile(fname_critic_optim):
            self.critic_optimizer.load_state_dict(torch.load(
                fname_critic_optim,
                map_location=self.device)
            )

    def declare_networks(self):
        pass

    def declare_memory(self):
        if self.priority_replay:
            self.memory = Memory(self.experience_replay_size)
        else:
            self.memory = MemoryDeque(self.experience_replay_size)

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

        shape = (-1,) + self.num_feats
        batch_state = torch.tensor(
            np.array(batch_state),
            device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(
            batch_action, device=self.device,
            dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(
            batch_reward, device=self.device,
            dtype=torch.float).squeeze().view(-1, 1)

        if weights is not None:
            torch.tensor(weights, device=self.device)

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

        Qvals = self.critic.forward(batch_state, batch_action)
        with torch.no_grad():
            next_Q = torch.zeros(
                self.batch_size, device=self.device,
                dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_actions = self.target_actor.forward(non_final_next_states)
                next_Q[non_final_mask] = self.target_critic.forward(
                    non_final_next_states, next_actions.detach())
            Qprime = batch_reward + self.gamma * next_Q

        critic_loss = self.critic_criterion(Qvals, Qprime)
        policy_loss = - \
            self.critic.forward(
                batch_state, self.actor.forward(batch_state)).mean()

        if self.priority_replay:
            self.memory.batch_update(
                indices, policy_loss.detach().squeeze().abs().cpu().numpy())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return critic_loss, policy_loss

    def update(self, s, a, r, s_, frame=0, store=True):
        if self.static_policy:
            return None

        if store:
            self.append_to_replay(s, a, r, s_)

        if self.priority_replay:
            data = [x for x in self.memory.tree.data if isinstance(
                x, int) and x != 0]
            if len(data) < self.batch_size:
                return None
        elif len(self.memory) < self.batch_size:
            return None

        batch_vars = self.prep_minibatch()

        critic_loss, policy_loss = self.compute_loss(batch_vars)
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
        self.save_sigma_param_magnitudes(frame)
        self.critic_losses.append(critic_loss)
        self.policy_losses.append(policy_loss)

    def get_action(self, s):
        with torch.no_grad():
            state = Variable(torch.from_numpy(s).float().unsqueeze(0))
            state = state.to(self.device)
            action = self.actor.forward(state)
            action = action.detach().cpu().numpy()[0, 0]
            return action

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma**i)
                     for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.store((state, action, R, None))

    def reset_hx(self):
        pass
