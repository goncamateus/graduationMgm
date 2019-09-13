from collections import deque  # Ordered collection with ends

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from graduationmgm.lib.DQN_train import DQNTrain


class DQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_outputs

        self.mlp1 = nn.Linear(*input_shape, 32)
        self.mlp2 = nn.Linear(32, 64)
        self.mlp3 = nn.Linear(64, 64)

        self.mlp4 = nn.Linear(self.feature_size(), 512)
        self.mlp5 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = x.view(x.size(0), -1)
        out = F.relu(self.mlp4(x))
        out = self.mlp5(out)
        return out

    def feature_size(self):
        x = self.mlp1(torch.zeros(1, *self.input_shape))
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x.view(1, -1).size(1)


class Model(DQNTrain):
    def __init__(self, static_policy=False, env=None, config=None):
        self.stacked_frames = deque(
            [np.zeros(env.observation_space.shape, dtype=np.int)
             for i in range(8)], maxlen=8)
        self.env = env
        self.num_feats = self.env.observation_space.shape
        super(Model, self).__init__(static_policy, env, config)

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.env.action_space.n)

    def stack_frames(self, frame, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frams
            self.stacked_frames = deque([np.zeros(
                frame.shape,
                dtype=np.int) for i in range(8)], maxlen=8)

            # Because we're in a new episode, copy the same frame 4x
            for _ in range(8):
                self.stacked_frames.append(frame)

            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=1)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different
            # frames)
            stacked_state = np.stack(self.stacked_frames, axis=1)

        return stacked_state
