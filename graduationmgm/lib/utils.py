import datetime
import logging
import pickle
import random
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class MemoryDeque():
    def __init__(self, length):
        self.mem = deque(maxlen=length)

    def store(self, mem):
        self.mem.append(mem)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.mem))
        return random.sample(self.mem, batch_size)

    def __len__(self):
        return len(self.mem)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0,
                 theta=0.15, max_sigma=0.3,
                 min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = 0
        self.high = 1
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def gen_mem_end(gen_mem, episode, model, frame_idx):
    gen_mem = False
    frame_idx = 0
    model.learn_start = 0
    logging.info('Start Learning at Episode %s', episode)


def episode_end(total_reward, episode, model):
    # Get the total reward of the episode
    logging.info('Episode %s reward %d', episode, total_reward)
    model.finish_nstep()
    model.reset_hx()


def save_modelmem(episode, test, model, model_path,
                  optim_path, mem_path, frame_idx, replay_size):
    if episode % 100 == 0 and episode > 0 and not test:
        model.save_w(path_model=model_path,
                     path_optim=optim_path)
    if ((frame_idx / 16) % replay_size) == 0 and episode % 1000 == 0:
        model.save_replay(mem_path=mem_path)
        logging.info("Memory Saved")


def save_rewards(rewards):
    day = datetime.datetime.now().today().day()
    hour = datetime.datetime.now().hour()
    minute = datetime.datetime.now().minute()
    final_str = str(day) + "-" + str(hour) + "-" + str(minute)
    with open('saved_agents/rewards_{}.pickle'.format(final_str),
              'w+') as fiile:
        pickle.dump(rewards, fiile)
        fiile.close()
