import datetime
import itertools
import logging
import math
import os
import pickle

import hfo
import numpy as np
import torch

from lib.hfo_env import HFOEnv
from lib.hyperparameters import Config


class Agent():

    config = Config()
    logging = logging.getLogger('Agent')

    def __init__(self, model, per):
        self.config_hyper(per)
        self.config_env()
        self.config_model(model)

    def config_hyper(self, per):
        # epsilon variables
        self.config.epsilon_start = 1.0
        self.config.epsilon_final = 0.01
        self.config.epsilon_decay = 30000
        self.config.epsilon_by_frame = lambda frame_idx: self.config.epsilon_final + \
            (self.config.epsilon_start - self.config.epsilon_final) * \
            math.exp(-1. * frame_idx / self.config.epsilon_decay)

        # misc agent variables
        self.config.GAMMA = 0.95
        self.config.LR = 0.00025
        # memory
        self.config.TARGET_NET_UPDATE_FREQ = 1000
        self.config.EXP_REPLAY_SIZE = 100000
        self.config.BATCH_SIZE = 64
        self.config.PRIORITY_ALPHA = 0.6
        self.config.PRIORITY_BETA_START = 0.4
        self.config.PRIORITY_BETA_FRAMES = 100000
        self.config.USE_PRIORITY_REPLAY = per

        # epsilon variables
        self.config.SIGMA_INIT = 0.5

        # Learning control variables
        self.config.LEARN_START = 100000
        self.config.MAX_FRAMES = 60000000
        self.config.UPDATE_FREQ = 1

        # Nstep controls
        self.config.N_STEPS = 1
        self.config.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def config_env(self):
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL]
        self.rewards = [700, 1000]
        self.hfo_env = HFOEnv(self.actions, self.rewards, strict=True)
        self.test = False
        self.gen_mem = True
        self.unum = self.hfo_env.getUnum()

    def config_model(self, model):

        self.dqn = model(env=self.hfo_env, config=self.config,
                         static_policy=self.test)
        self.currun_rewards = list()
        self.model_path = './saved_agents/model_{}.dump'.format(self.unum)
        self.optim_path = './saved_agents/optim_{}.dump'.format(self.unum)
        self.mem_path = './saved_agents/exp_replay_agent_{}.dump'.format(
            self.unum)

        if os.path.isfile(self.model_path) and os.path.isfile(self.optim_path):
            self.dqn.load_w(model_path=self.model_path,
                            optim_path=self.optim_path)
            self.logging.info("Model Loaded")

        if not self.test:
            if os.path.isfile(self.mem_path):
                self.dqn.load_replay(mem_path=self.mem_path)
                self.dqn.learn_start = 0
                self.logging.info("Memory Loaded")

    def gen_mem_end(self, episode):
        self.gen_mem = False
        self.frame_idx = 0
        self.dqn.learn_start = 0
        self.logging.info('Start Learning at Episode %s', episode)

    def episode_end(self, total_reward, episode, state, frame):
        # Get the total reward of the episode
        self.logging.info('Episode %s reward %d', episode, total_reward)
        self.dqn.finish_nstep()
        self.dqn.reset_hx()
        # We finished the episode
        next_state = np.zeros(state.shape)
        next_frame = np.zeros(frame.shape)

    def save_modelmem(self, episode):
        if episode % 100 == 0 and episode > 0 and not self.test:
            self.dqn.save_w(path_model=self.model_path,
                            path_optim=self.optim_path)
        if ((self.frame_idx / 16) % self.config.EXP_REPLAY_SIZE) == 0 and episode % 1000 == 0:
            self.dqn.save_replay(mem_path=self.mem_path)
            self.logging.info("Memory Saved")

    def save_rewards(self):
        day = datetime.datetime.now().today().day()
        hour = datetime.datetime.now().hour()
        minute = datetime.datetime.now().minute()
        final_str = str(day) + "-" + str(hour) + "-" + str(minute)
        with open('saved_agents/rewards_{}.pickle'.format(final_str),
                  'w+') as fiile:
            pickle.dump(self.currun_rewards, fiile)
            fiile.close()

    def bye(self, status):
        if status == hfo.SERVER_DOWN:
            if not self.test:
                self.dqn.save_w(path_model=self.model_path,
                                path_optim=self.optim_path)
                print("Model Saved")
                self.dqn.save_replay(mem_path=self.mem_path)
                print("Memory Saved")
            self.save_rewards()
            self.hfo_env.act(hfo.QUIT)
            exit()

    def run(self):
        self.frame_idx = 1
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = list()
            while status == hfo.IN_GAME:
                # Every time when game resets starts a zero frame
                if done:
                    state = self.hfo_env.get_state(strict=True)
                    frame = self.dqn.stack_frames(state, done)
                # If the size of experiences is under max_size*16 runs gen_mem
                # Biasing the agent for the Agent2d Helios_base
                if self.gen_mem and self.frame_idx / 16 < self.config.EXP_REPLAY_SIZE:
                    action = 1 if self.hfo_env.isIntercept() else 0
                else:
                    # When gen_mem is done, saves experiences and starts a new
                    # frame counting and starts the learning process
                    if self.gen_mem:
                        self.gen_mem_end(episode)

                    # Calculates epsilon on frame according to the stack index
                    # and gets the action
                    epsilon = self.config.epsilon_by_frame(int(self.frame_idx / 16))
                    action = self.dqn.get_action(frame, epsilon)

                # Calculates results from environment
                next_state, reward, done, status = self.hfo_env.step(action,
                                                                strict=True)
                episode_rewards.append(reward)

                if done:
                    # Resets frame_stack and states
                    total_reward = np.sum(episode_rewards)
                    self.currun_rewards.append(total_reward)
                    self.episode_end(total_reward, episode, state, frame)
                else:
                    next_frame = self.dqn.stack_frames(next_state, done)

                self.dqn.update(frame, action, reward,
                                next_frame, int(self.frame_idx / 16))
                frame = next_frame
                state = next_state

                self.frame_idx += 1
                self.save_modelmem(episode)

            self.bye(status)
