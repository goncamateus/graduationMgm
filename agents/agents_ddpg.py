import datetime
import gc
import itertools
import logging
import os
import pickle
import time

import hfo
import numpy as np
from gym import spaces
from joblib import Parallel, delayed

from agents.base_agent import Agent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.utils import AsyncWrite, MemoryDeque

logger = logging.getLogger('Agent')


class ObservationSpace():
    def __init__(self, shape=None):
        self.nmr_out = 0
        self.taken = 0
        self.shape = shape
        self.goals_taken = 0


class ActionSpaceContinuous(spaces.Box):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        super().__init__(low, high, shape=shape, dtype=dtype)


class MockEnv:
    def __init__(self, num_mates, num_ops):
        self.action_space = ActionSpaceContinuous(-1, 1, shape=(1,))
        shape = 11 + 3 * num_mates + 2 * num_ops
        shape = (shape,)
        self.observation_space = ObservationSpace(shape=shape)
        self.continuous = True


class DDPGAgent(Agent):

    memory_save_thread = None

    def __init__(self, model, per, team='base', port=6000, num_agents=1, num_ops=1):
        self.num_agents = num_agents
        self.num_ops = num_ops
        self.unums = list()
        self.goals = 0
        self.team = team
        self.port = port
        self.config_env()
        self.config_hyper(per)
        self.config_model(model)

    def config_env(self):
        BLOCK = hfo.CATCH
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL, BLOCK]
        self.rewards = [0, 0, 0]
        self.test = False
        self.gen_mem = True

    def load_model(self, model):
        env = MockEnv(self.num_agents, self.num_ops)
        self.ddpg = model(env=env, config=self.config,
                          static_policy=self.test)
        self.model_paths = (f'./saved_agents/DDPG/actor.dump',
                            f'./saved_agents/DDPG/critic.dump')
        self.optim_paths = (f'./saved_agents/DDPG/actor_optim.dump',
                            f'./saved_agents/DDPG/critic_optim.dump')
        if os.path.isfile(self.model_paths[0]) \
                and os.path.isfile(self.optim_paths[0]):
            self.ddpg.load_w(path_models=self.model_paths,
                             path_optims=self.optim_paths)
            print("Model Loaded")
        # self.ddpg.share_memory()

    def load_memory(self):
        self.mem_path = 'exp_replay_agent_'
        paths = ['./saved_agents/DDPG/' +
                 x for x in os.listdir('./saved_agents/DDPG/') if self.mem_path in x]
        memories = list()
        for path in paths:
            if os.path.isfile(path) and not self.test:
                self.ddpg.load_replay(mem_path=path)
                if len(self.ddpg.memory) >= self.config.EXP_REPLAY_SIZE:
                    self.gen_mem_end(0)
                memories.append(self.ddpg.memory)
        if memories:
            self.ddpg.memory = memories
            self.gen_mem = False
            print("Memory Loaded")

    def save_model(self, episode=0, bye=False):
        saved = False
        if (episode % 100 == 0 and episode > 0) or bye:
            self.ddpg.save_w(path_models=self.model_paths,
                             path_optims=self.optim_paths)
            print("Model Saved")
            saved = True
        return saved

    def save_mem(self, episode=0, bye=False):
        saved = False
        if episode % 5 == 0 and self.memory_save_thread is not None:
            self.memory_save_thread.join()
            self.memory_save_thread = None
        if (episode % 1000 == 0 and episode > 2 and not self.test) or bye:
            self.memory_save_thread = AsyncWrite(
                self.ddpg.memory, self.mem_path, 'Memory saved')
            self.memory_save_thread.start()
            saved = True
        return saved

    def save_loss(self, episode=0, bye=False):
        losses = (self.ddpg.critic_loss, self.ddpg.actor_loss)
        if (episode % 100 == 0 and episode > 0 and not self.test) or bye:
            with open(f'./saved_agents/{self.ddpg.__name__}/{self.ddpg.__name__}.loss', 'wb') as lf:
                pickle.dump(losses, lf)
                lf.close()

    def save_rewards(self, episode=0, bye=False):
        if (episode % 100 == 0 and episode > 0 and not self.test) or bye:
            with open(f'./saved_agents/{self.ddpg.__name__}/{self.ddpg.__name__}.reward', 'wb') as lf:
                pickle.dump(self.currun_rewards, lf)
                lf.close()
