import datetime
import gc
import itertools
import logging
import os
import pickle
import time
from joblib import delayed, Parallel
from copy import deepcopy
# from concurrent.futures import ProcessPoolExecutor
from functools import partial

import hfo
import numpy as np

from agents.base_agent import Agent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.utils import AsyncWrite, OUNoise

logger = logging.getLogger('Agent')


def init_env(port, team, actions, rewards, num_agent):
    time.sleep(num_agent*5)
    env = HFOEnv()
    env.connect(is_offensive=False, play_goalie=False,
                port=port, continuous=True,
                team=team)
    while not env.waitAnyState():
        pass
    while not env.waitToAct():
        pass
    assert env.processBeforeBegins()
    env.set_env(actions, rewards, strict=True)
    return env


def set_env(actions, rewards, env):
    return env.getUnum()


class DDPGAgent(Agent):

    memory_save_thread = None

    def __init__(self, model, per, team='base', port=6000, num_agents=1):
        self.num_agents = num_agents
        self.unums = list()
        self.goals = 0
        self.team = team
        self.port = port
        self.init_envs()
        # self.wait_ready()
        # self.set_envs()
        self.config_hyper(per)
        self.config_model(model)
        self.run()

    def init_envs(self):
        BLOCK = hfo.CATCH
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL, BLOCK]
        self.rewards = [0, 0, 0]
        self.envs = list()
        self.test = False
        self.gen_mem = True
        init_environs = partial(init_env, self.port,
                                self.team, self.actions, self.rewards)
        self.envs = Parallel(n_jobs=self.num_agents)(
            delayed(init_environs)(num) for num in range(self.num_agents))
        # for i in range(self.num_agents):
        #     env = HFOEnv()
        #     self.envs.append(env)
        #     env.connect(is_offensive=False, play_goalie=False,
        #                 port=self.port, continuous=True,
        #                 team=self.team)

    def wait_ready(self):
        any_state = [False for _ in range(self.num_agents)]
        ready = [False for _ in range(self.num_agents)]
        while False in ready:
            for i, env in enumerate(self.envs):
                if not any_state[i]:
                    any_state[i] = env.waitAnyState()
                if any_state[i] and not ready[i]:
                    ready[i] = env.waitToAct()
                    if ready[i]:
                        assert env.processBeforeBegins()

    def set_envs(self):
        # with ProcessPoolExecutor() as executor:
        #     self.envs = executor.map(
        #         init_environs,
        #         range(self.num_agents)
        #     )
        set_environs = partial(set_env, self.actions, self.rewards)
        self.unums = Parallel(n_jobs=self.num_agents)(delayed(set_environs)(env)
                                                      for env in self.envs)
        self.unums = list(self.unums)
        # for env in self.envs:
        #     env.set_env(self.actions, self.rewards, strict=True)
        #     self.unums.append(env.getUnum())

    def load_model(self, model):
        print(self.envs[0].getUnum())
        self.ddpg = model(env=self.envs[0], config=self.config,
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

    def load_memory(self):
        self.mem_path = f'./saved_agents/DDPG/exp_replay_agent_ddpg.dump'
        if os.path.isfile(self.mem_path) and not self.test:
            self.ddpg.load_replay(mem_path=self.mem_path)
            self.gen_mem_end(0)
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
            # self.ddpg.save_replay(mem_path=self.mem_path)
            # print('Memory saved')
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

    def run(self):
        print('CHEGOU AQUI')
        self.frame_idx = 1
        self.goals = 0
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = 0
            step = 0
            while status == hfo.IN_GAME:
                for env in self.envs:
                    # Every time when game resets starts a zero frame
                    state = env.get_state()
                    # interceptable = state[-1]
                    frame = self.ddpg.stack_frames(state, done)
                    # If the size of experiences is under max_size*8 runs gen_mem
                    if self.gen_mem and len(self.ddpg.memory) < self.config.EXP_REPLAY_SIZE:
                        action = env.action_space.sample()
                    else:
                        # When gen_mem is done, saves experiences and starts a new
                        # frame counting and starts the learning process
                        if self.gen_mem:
                            self.gen_mem_end(episode)
                        # Gets the action
                        action = self.ddpg.get_action(frame)
                        action = (action + np.random.normal(0, 0.1, size=env.action_space.shape[0])).clip(
                            env.action_space.low, env.action_space.high)
                        action = action.astype(np.float32)
                        step += 1

                    # if interceptable and self.gen_mem:
                    #     action = np.array(
                    #         [np.random.uniform(-0.68, 0.36)], dtype=np.float32)
                    #     action = (action + np.random.normal(0, 0.1, size=self.hfo_env.action_space.shape[0])).clip(
                    #         self.hfo_env.action_space.low, self.hfo_env.action_space.high)
                    #     action = action.astype(np.float32)

                    # Calculates results from environment
                    next_state, reward, done, status = env.step(action)
                    episode_rewards += reward

                    if done:
                        # Resets frame_stack and states
                        # if not self.gen_mem:
                        #     self.ddpg.writer.add_scalar(
                        #         f'Rewards/epi_reward', episode_rewards, global_step=episode)
                        # self.currun_rewards.append(episode_rewards)
                        next_state = np.zeros(state.shape)
                        next_frame = np.zeros(frame.shape)
                        if episode % 100 == 0 and episode > 10 and self.goals > 0:
                            if env.getUnum() == self.unums[0]:
                                print(self.goals)
                                self.goals = 0
                    else:
                        next_frame = self.ddpg.stack_frames(next_state, done)

                    self.ddpg.append_to_replay(
                        frame, action, reward, next_frame, int(done))
                if status == hfo.GOAL:
                    self.goals += 1
                if not self.gen_mem and not self.test:
                    self.ddpg.update()
                if done:
                    break
                self.frame_idx += 1
            if not self.gen_mem or self.test:
                self.save_modelmem(episode)
            gc.collect()
            self.bye(status)
