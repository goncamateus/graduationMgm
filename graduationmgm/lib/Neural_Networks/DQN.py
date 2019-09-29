from collections import deque  # Ordered collection with ends

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graduationmgm.lib.DQN_train import DQNTrain


class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs),
        )

    def forward(self, x):
        x = self.feature(x)
        return x


class Model(DQNTrain):
    def __init__(self, static_policy=False, env=None, config=None):
        self.stacked_frames = deque(
            [np.zeros(env.observation_space.shape[0], dtype=np.int)
             for i in range(8)], maxlen=8)
        self.env = env
        self.num_feats = (self.env.observation_space.shape[0]*8,)
        super(Model, self).__init__(static_policy, env, config)

    def declare_networks(self):
        self.model = DQN(self.env.observation_space.shape[0]*8,
                         self.env.action_space.n)

    def stack_frames(self, state, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frams
            self.stacked_frames = deque([np.zeros(
                state.shape,
                dtype=np.int) for i in range(8)], maxlen=8)

            # Because we're in a new episode, copy the same state 8x
            for _ in range(8):
                self.stacked_frames.append(state)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=0)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(state)

            # Build the stacked state (first dimension specifies different
            # frames)
            stacked_state = np.stack(self.stacked_frames, axis=0)

        return stacked_state
