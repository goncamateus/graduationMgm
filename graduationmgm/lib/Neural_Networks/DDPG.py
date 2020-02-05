from collections import deque  # Ordered collection with ends

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graduationmgm.lib.DDPG_train import DDPGTrain


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)

        self.max_action = max_action

    def forward(self, x):
        try:
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))
        except Exception as e:
            print(e)
            raise e
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x, u):
        try:
            x = F.relu(self.l1(torch.cat([x, u], 1)))
            x = F.relu(self.l2(x))
            x = self.l3(x)
        except Exception as e:
            print(e)
            raise e
        return x


class DDPG(DDPGTrain):

    __name__ = 'DDPG'

    def __init__(self, static_policy=False, env=None, config=None):
        self.stack_size = 128
        self.stacked_frames = deque(
            [np.zeros(env.observation_space.shape, dtype=np.int)
             for i in range(self.stack_size)], maxlen=self.stack_size)
        self.max_action = float(env.action_space.high[0])
        super(DDPG, self).__init__(static_policy, env, config)
        self.num_feats = (*self.env.observation_space.shape,
                          len(self.stacked_frames))

    def declare_networks(self):
        self.actor = Actor(self.env.observation_space.shape[0]*self.stack_size,
                           self.env.action_space.shape[0], self.max_action)
        self.target_actor = Actor(self.env.observation_space.shape[0]*self.stack_size,
                                  self.env.action_space.shape[0], self.max_action)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            self.env.observation_space.shape[0]*self.stack_size,
            self.env.action_space.shape[0])
        self.target_critic = Critic(
            self.env.observation_space.shape[0]*self.stack_size,
            self.env.action_space.shape[0])
        self.target_critic.load_state_dict(self.critic.state_dict())

    def share_memory(self):
        self.actor.share_memory()
        self.target_actor.share_memory()
        self.critic.share_memory()
        self.target_critic.share_memory()

    def stack_frames(self, frame, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frams
            self.stacked_frames = deque([np.zeros(
                frame.shape,
                dtype=np.int) for i in range(self.stack_size)], maxlen=self.stack_size)

            # Because we're in a new episode, copy the same frame self.stack_sizex
            for _ in range(self.stack_size):
                self.stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=0)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different
            # frames)
            stacked_state = np.stack(self.stacked_frames, axis=0)

        return stacked_state
