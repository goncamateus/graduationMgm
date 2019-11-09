import os.path

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from graduationmgm.lib.base_train import BaseTrain
from graduationmgm.lib.utils import MemoryDeque

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()


class DDPGTrain(BaseTrain):
    def __init__(self, static_policy=False, env=None,
                 config=None):
        super(DDPGTrain, self).__init__(config=config, env=env)

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
        self.env = env
        self.writer = SummaryWriter(
            f'./saved_agents/DDPG/agent_{self.env.getUnum()}')
        self.declare_networks()
        actor_learning_rate = 1e-4
        critic_learning_rate = 1e-3
        self.num_actor_update_iteration = 0
        self.num_critic_update_iteration = 0
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate)
        self.actor_loss = self.critic_loss = list()

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
        self.memory = MemoryDeque(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_, d):
        self.memory.store((s, a, r, s_, d))

    def update(self):  # faster
        for _ in range(100):
            state, next_state, action, reward, done = self.memory.sample(
                self.batch_size)
            reward = reward.reshape(-1, 1)
            done = done.reshape(-1, 1)
            num_feat = state.shape[1] * state.shape[2]
            state = Variable(torch.FloatTensor(
                np.float32(state))).view(self.batch_size, num_feat)
            next_state = Variable(torch.FloatTensor(
                np.float32(next_state))).view(self.batch_size, num_feat)
            action = Variable(torch.FloatTensor(action))
            reward = Variable(torch.FloatTensor(reward))
            done = Variable(torch.FloatTensor(done))

            # Compute the target Q value
            acts = self.target_actor(next_state)
            target_Q = self.target_critic(next_state, acts)
            target_Q = reward + (self.gamma * target_Q * (1 - done)).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.smooth_l1_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.writer.add_scalar('Loss/ddpg/critic_loss', critic_loss,
                                global_step=self.num_critic_update_iteration)
            self.critic_loss.append(critic_loss)

            # Compute actor loss
            acts = self.actor(state)
            actor_loss = -self.critic(state, acts).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.writer.add_scalar(
                'Loss/ddpg/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            self.actor_loss.append(actor_loss)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


    def get_action(self, s):
        with torch.no_grad():
            num_feat = s.shape[0] * s.shape[1]
            state = Variable(torch.from_numpy(s).float().unsqueeze(0))
            state = state.view(1, num_feat)
            action = self.actor.forward(state)
            action = action.detach().cpu().numpy()[0, 0]
            return action
