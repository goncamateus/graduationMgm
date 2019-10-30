import os.path
import pickle

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from graduationmgm.lib.base_train import BaseTrain
from graduationmgm.lib.prioritized_experience_replay import Memory
from graduationmgm.lib.utils import MemoryDeque

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda()


class DQNTrain(BaseTrain):
    def __init__(self, static_policy=False, env=None,
                 config=None, log_dir='/tmp/RC_test'):
        super(DQNTrain, self).__init__(
            config=config, env=env, log_dir=log_dir)
        self.noisy = config.USE_NOISY_NETS
        self.priority_replay = config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
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
        self.writer = SummaryWriter(
            f'./saved_agents/DQN/agent_{self.env.getUnum()}')
        self.losses = list()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)

        if self.static_policy:
            self.model.eval()
        else:
            self.model.train()

        self.update_iteration = 0

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

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(
                torch.load(fname_optim, map_location=self.device))

    def declare_networks(self):
        pass

    def declare_memory(self):
        self.memory = MemoryDeque(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_, d):
        self.memory.store((s, a, r, s_, d))

    def compute_td_loss(self):
        state, next_state, action, reward, done = self.memory.sample(
            self.batch_size)
        num_feat = state.shape[1] * state.shape[2]
        state = Variable(torch.FloatTensor(
            np.float32(state))).view(64, num_feat)
        next_state = Variable(torch.FloatTensor(
            np.float32(next_state))).view(64, num_feat)
        action = Variable(torch.LongTensor(action))
        reward = Variable(torch.FloatTensor(reward))
        done = Variable(torch.FloatTensor(done))

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss = F.smooth_l1_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update(self, frame=0):
        loss = self.compute_td_loss()
        unum = self.env.getUnum()
        self.writer.add_scalar(
            f'Loss/dqn/loss_{unum}', loss, global_step=self.update_iteration)
        self.losses.append(loss)
        self.update_iteration += 1

    def get_action(self, s, eps=0.1):  # faster
        with torch.no_grad():
            if np.random.uniform() >= eps or self.static_policy or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.view(1, -1)
                out = self.model(X)
                maxout = out.argmax()
                return maxout.item()
            else:
                return np.random.randint(0, self.num_actions)
