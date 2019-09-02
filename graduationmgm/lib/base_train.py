import csv
import os.path
import pickle

import numpy as np
import torch
import torch.optim as optim


class BaseTrain(object):
    def __init__(self, config, env, log_dir='/tmp/RC_test'):
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.device = config.device
        self.log_dir = log_dir
        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None        

        self.rewards = []

        self.action_log_frequency = config.ACTION_SELECTION_COUNT_FREQUENCY
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

    def save_sigma_param_magnitudes(self, tstep):
        with torch.no_grad():
            sum_, count = 0.0, 0.0
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'sigma' in name:
                    sum_ += torch.sum(param.abs()).item()
                    count += np.prod(param.shape)
        if count > 0:
            with open(os.path.join(self.log_dir,
                                   'sig_param_mag.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow((tstep, sum_ / count))

    def save_td(self, td, tstep):
        with open(os.path.join('./saved_agents', 'td.csv'), 'w+') as f:
            writer = csv.writer(f)
            writer.writerow((tstep, td))

    def save_reward(self, reward):
        self.rewards.append(reward)

    def save_action(self, action, tstep):
        self.action_selections[int(action)] += 1.0 / self.action_log_frequency
        if (tstep + 1) % self.action_log_frequency == 0:
            with open(os.path.join(self.log_dir, 'action_log.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list([tstep] + self.action_selections))
            self.action_selections = [
                0 for _ in range(len(self.action_selections))]
