from collections import deque  # Ordered collection with ends

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graduationmgm.lib.DDPG_train import DDPGTrain


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = action_dim

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Model(DDPGTrain):
    def __init__(self, static_policy=False, env=None, config=None):
        self.stacked_frames = deque(
            [np.zeros(env.observation_space.shape, dtype=np.int)
             for i in range(8)], maxlen=8)
        super(Model, self).__init__(static_policy, env, config)
        self.num_feats = (*self.env.observation_space.shape,
                          len(self.stacked_frames))

    def declare_networks(self):
        self.actor = Actor(
            self.env.observation_space.shape[0]*8, self.env.action_space.shape[0])
        self.target_actor = Actor(
            self.env.observation_space.shape[0]*8, self.env.action_space.shape[0])
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            self.env.observation_space.shape[0]*8,
            self.env.action_space.shape[0])
        self.target_critic = Critic(
            self.env.observation_space.shape[0]*8,
            self.env.action_space.shape[0])
        self.target_critic.load_state_dict(self.critic.state_dict())

    def stack_frames(self, frame, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frams
            self.stacked_frames = deque([np.zeros(
                frame.shape,
                dtype=np.int) for i in range(8)], maxlen=8)

            # Because we're in a new episode, copy the same frame 8x
            for _ in range(8):
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
