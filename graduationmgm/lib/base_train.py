import csv
import os.path
import pickle

import numpy as np
import torch


class BaseTrain(object):
    def __init__(self, config, env):
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.device = config.device
        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None

        self.rewards = []

        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY
        if not env.continuous:
            self.action_selections = [0 for _ in range(env.action_space.n)]

    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

    def MSE(self, x):
        return 0.5 * x.pow(2)

    def save_w(self, path_model=None, path_optim=None):
        if path_model is None:
            torch.save(self.model.state_dict(), './saved_agents/model.dump')
        else:
            torch.save(self.model.state_dict(), path_model)
        if path_optim is None:
            torch.save(self.optimizer.state_dict(),
                       './saved_agents/optim.dump')
        else:
            torch.save(self.optimizer.state_dict(), path_optim)

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
            self.model.load_state_dict(torch.load(fname_model,
                                                  map_location=self.device))

        if os.path.isfile(fname_optim):
            self.optimizer.load_state_dict(torch.load(fname_optim,
                                                      map_location=self.device)
                                           )

    def save_replay(self, mem_path='./saved_agents/exp_replay_agent.dump'):
        pickle.dump(self.memory, open(mem_path, 'wb'))

    def load_replay(self, mem_path='./saved_agents/exp_replay_agent.dump'):
        if os.path.isfile(mem_path):
            self.memory = pickle.load(open(mem_path, 'rb'))
