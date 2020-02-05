import datetime
import itertools
import logging
import math
import os
import pickle

import hfo
import numpy as np
import torch
from tensorboardX import SummaryWriter

from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.hyperparameters import Config

logger = logging.getLogger('Agent')


class Agent():

    config = Config()

    def __init__(self, model, per, team='base', port=6000):
        self.config_env(team, port)
        self.config_hyper(per)
        self.config_model(model)
        self.episodes = 10000
        self.goals = 0

    def config_hyper(self, per):
        # epsilon variables
        self.config.epsilon_start = 0.01
        self.config.epsilon_final = 0.01
        self.config.epsilon_decay = 30000
        self.config.epsilon_by_frame = lambda frame_idx: self.config.epsilon_final + \
            (self.config.epsilon_start - self.config.epsilon_final) * \
            math.exp(-1. * frame_idx / self.config.epsilon_decay)

        # misc agent variables
        self.config.GAMMA = 0.95
        self.config.LR = 0.00025
        # memory
        self.config.USE_PRIORITY_REPLAY = per
        self.config.TARGET_NET_UPDATE_FREQ = 1000
        self.config.EXP_REPLAY_SIZE = 3e5
        self.config.BATCH_SIZE = 64

        # Learning control variables
        self.config.LEARN_START = 300000
        self.config.MAX_FRAMES = 60000000
        self.config.UPDATE_FREQ = 1

        # Nstep controls
        self.config.N_STEPS = 1
        # if not self.test:
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # else:
        # self.config.device = torch.device("cpu")

    def config_env(self, team, port):
        BLOCK = hfo.CATCH
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL, hfo.DEFEND_GOAL, BLOCK]
        self.rewards = [0, 0, 0, 0]
        self.hfo_env = HFOEnv(self.actions, self.rewards,
                              strict=True, port=port, team=team)
        self.test = False
        self.gen_mem = True
        self.unum = self.hfo_env.getUnum()

    def load_model(self, model):
        self.dqn = model(env=self.hfo_env, config=self.config,
                         static_policy=self.test)
        self.model_path = './saved_agents/{}/model_{}.dump'.format(
            self.dqn.__name__, self.unum)
        self.optim_path = './saved_agents/{}/optim_{}.dump'.format(
            self.dqn.__name__, self.unum)
        if os.path.isfile(self.model_path) and os.path.isfile(self.optim_path):
            self.dqn.load_w(model_path=self.model_path,
                            optim_path=self.optim_path)
            print("Model Loaded")

    def load_memory(self):
        self.mem_path = './saved_agents/{}/exp_replay_agent_{}.dump'.format(
            self.dqn.__name__, self.unum)

        if not self.test:
            if os.path.isfile(self.mem_path):
                self.dqn.load_replay(mem_path=self.mem_path)
                self.gen_mem_end(0)
                print("Memory Loaded")

    def save_model(self, episode=0, bye=False):
        saved = False
        if (episode % 100 == 0 and episode > 0) or bye:
            self.dqn.save_w(path_model=self.model_path,
                            path_optim=self.optim_path)
            print("Model Saved")
            saved = True
        return saved

    def save_mem(self, episode=0, bye=False):
        saved = False
        if (episode % 1000 == 0 and episode > 2 and not self.test) or bye:
            self.dqn.save_replay(mem_path=self.mem_path)
            print("Memory Saved")
            saved = True
        return saved

    def save_loss(self, episode=0, bye=False):
        if (episode % 100 == 0 and episode > 0 and not self.test) or bye:
            with open(f'./saved_agents/{self.dqn.__name__}/{self.dqn.__name__}.loss', 'wb') as lf:
                pickle.dump(self.dqn.losses, lf)
                lf.close()

    def save_rewards(self, episode=0, bye=False):
        if (episode % 100 == 0 and episode > 0 and not self.test) or bye:
            with open(f'./saved_agents/{self.dqn.__name__}/{self.dqn.__name__}.reward', 'wb') as lf:
                pickle.dump(self.currun_rewards, lf)
                lf.close()

    def config_model(self, model):
        try:
            os.mkdir(f'saved_agents/{model.__name__}')
        except FileExistsError:
            pass
        self.load_model(model)
        self.load_memory()
        self.currun_rewards = list()

    def gen_mem_end(self, episode):
        self.gen_mem = False
        self.frame_idx = 0
        print('Start Learning at Episode %s', episode)

    def save_modelmem(self, episode=0, bye=False):
        s1 = self.save_model(episode, bye)
        s2 = self.save_mem(episode, bye)
        return s1 or s2

    def bye(self, status=hfo.SERVER_DOWN, thread=None):
        if status == hfo.SERVER_DOWN:
            if thread:
                thread.join()
            self.hfo_env.act(hfo.QUIT)
            exit()

    def run(self):
        self.frame_idx = 1
        self.goals = 0
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = 0
            while status == hfo.IN_GAME:
                # Every time when game resets starts a zero frame
                if done:
                    state_ori = self.hfo_env.get_state()
                    # interceptable = state_ori[-1]
                    state = state_ori
                    frame = self.dqn.stack_frames(state, done)
                # If the size of experiences is under max_size*8 runs gen_mem
                if self.gen_mem and len(self.dqn.memory) < self.config.EXP_REPLAY_SIZE:
                    action = np.random.randint(0, len(self.actions))
                else:
                    # When gen_mem is done, saves experiences and starts a new
                    # frame counting and starts the learning process
                    if self.gen_mem:
                        self.gen_mem_end(episode)

                    # Calculates epsilon on frame according to the stack index
                    # and gets the action
                    epsilon = self.config.epsilon_by_frame(
                        self.frame_idx)
                    action = self.dqn.get_action(frame, epsilon)
                # action = action if not interceptable else 1

                # Calculates results from environment
                next_state_ori, reward, done, status = self.hfo_env.step(
                    action)
                next_state = next_state_ori
                episode_rewards += reward

                if done:
                    # Resets frame_stack and states
                    if not self.gen_mem:
                        self.dqn.writer.add_scalar(
                            f'Rewards/{self.dqn.__name__}/epi_reward_{self.unum}', episode_rewards, global_step=episode)
                    if status == hfo.GOAL:
                        self.goals += 1
                    if episode % 100 == 0 and episode > 10 and self.goals > 0:
                        print(self.goals)
                        self.goals = 0
                    self.currun_rewards.append(episode_rewards)
                    next_state = np.zeros(state.shape)
                    next_frame = np.zeros(frame.shape)
                else:
                    next_frame = self.dqn.stack_frames(next_state, done)

                self.dqn.append_to_replay(
                    frame, action, reward, next_frame, int(done))
                frame = next_frame
                state = next_state
                if done:
                    break
                self.frame_idx += 1
            if not self.gen_mem or self.test:
                self.dqn.update(self.frame_idx)
            self.save_modelmem(episode)
            self.bye(status)
