import datetime
import gc
import itertools
import logging
import math
import os
import pickle
import random
from collections import deque
from functools import partial

import hfo
import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean

from agents.base_agent import Agent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.utils import AsyncWrite, OUNoise

logger = logging.getLogger('Agent')
np.random.seed(5)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def calc_dist(buffer, state):
    dist_sum = 0
    max_dist = -10e6
    for bstate in buffer:
        dist = euclidean(state, bstate)
        max_dist = abs(dist) if abs(dist) > max_dist else max_dist
        dist_sum += dist
    dist_sum = dist_sum/len(buffer)
    return dist_sum, max_dist


def calc_rewards_job(gbuffer, bbuffer, good, state):
    dist_success = dist_fail = 1
    max_success = max_fail = 1
    if gbuffer:
        dist_success, max_success = calc_dist(gbuffer, state)
    if bbuffer:
        dist_fail, max_fail = calc_dist(bbuffer, state)
    if good:
        reward = (abs(dist_fail) / max_fail) - \
            (abs(dist_success) / max_success)
        reward = np.exp(reward)
    else:
        reward = (abs(dist_success) / max_success) - \
            (abs(dist_fail) / max_fail)
        reward = -np.exp(reward)
    return reward


class DDPGAgent(Agent):

    memory_save_thread = None

    def __init__(self, model, per, team='base', port=6000):
        self.config_env(team, port)
        self.config_hyper(per)
        self.config_model(model)
        self.goals = 0

    def config_env(self, team, port):
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL, hfo.BLOCK]
        self.taken_action = [0, 0, 0]
        self.rewards = [0, 0, 0]
        self.hfo_env = HFOEnv(is_offensive=False, play_goalie=False,
                              port=port, continuous=True,
                              team=team)
        self.hfo_env.set_env(self.actions, self.rewards, strict=True)
        self.test = False
        self.gen_mem = True
        self.unum = self.hfo_env.getUnum()

    def load_model(self, model):
        self.ddpg = model(env=self.hfo_env, config=self.config,
                          static_policy=self.test)
        self.model_paths = (f'./saved_agents/DDPG/actor_{self.unum}_gexp.dump',
                            f'./saved_agents/DDPG/critic_{self.unum}_gexp.dump')
        self.optim_paths = (f'./saved_agents/DDPG/actor_optim_{self.unum}_gexp.dump',
                            f'./saved_agents/DDPG/critic_optim_{self.unum}_gexp.dump')
        if os.path.isfile(self.model_paths[0]) \
                and os.path.isfile(self.optim_paths[0]):
            self.ddpg.load_w(path_models=self.model_paths,
                             path_optims=self.optim_paths)
            print("Model Loaded")

    def load_memory(self):
        self.mem_path = f'./saved_agents/DDPG/exp_replay_agent_{self.unum}_gexp_ddpg.dump'
        if os.path.isfile(self.mem_path) and not self.test:
            self.ddpg.load_replay(mem_path=self.mem_path)
            self.gen_mem_end(0)
            self.config.epsilon_decay = 1
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

    def calc_dist(self, buffer, state):
        dist_sum = 0
        max_dist = -10e6
        for bstate in buffer:
            dist = euclidean(state, bstate)
            max_dist = abs(dist) if abs(dist) > max_dist else max_dist
            dist_sum += dist
        dist_sum = dist_sum/len(buffer)
        return dist_sum, max_dist

    def calc_rewards(self, gbuffer, bbuffer, ibuffer, good=True):
        buffer = list()
        for i, state in enumerate([x[0] for x in ibuffer]):
            dist_success = dist_fail = 1
            max_success = max_fail = 1
            if gbuffer:
                dist_success, max_success = self.calc_dist(gbuffer, state)
            if bbuffer:
                dist_fail, max_fail = self.calc_dist(bbuffer, state)
            if good:
                reward = (abs(dist_fail) / max_fail) - \
                    (abs(dist_success) / max_success)
                reward = np.exp(reward)
            else:
                reward = (abs(dist_success) / max_success) - \
                    (abs(dist_fail) / max_fail)
                reward = -np.exp(reward)
            reward = reward * np.exp(i/len(ibuffer)) / math.e
            buffer.append(
                (state, ibuffer[i][1], reward, ibuffer[i][3], ibuffer[i][4]))
        return buffer

    def calc_rewards_parallel(self, gbuffer, bbuffer, ibuffer, good=True):
        buffer = list()
        calc_job = partial(calc_rewards_job, gbuffer, bbuffer, good)
        states = [x[0] for x in ibuffer]
        rewards = Parallel(n_jobs=-1)(delayed(calc_job)(state)
                                      for state in states)
        for i in range(ibuffer):
            buffer.append(
                (ibuffer[i][0], ibuffer[i][1],
                 ibuffer[i][2] + rewards[i],
                 ibuffer[i][3], ibuffer[i][4]))
        return buffer

    def run(self):
        self.frame_idx = 1
        self.goals = 0
        good_buffer = deque([], maxlen=100)
        bad_buffer = deque([], maxlen=100)
        for episode in itertools.count():
            intermed_buffer = ibuffer = list()
            status = hfo.IN_GAME
            done = True
            step = 0
            while status == hfo.IN_GAME:
                # Every time when game resets starts a zero frame
                if done:
                    state_ori = self.hfo_env.get_state()
                    state = state_ori
                    frame = self.ddpg.stack_frames(state, done)
                # If the size of experiences is under max_size*8 runs gen_mem
                if self.gen_mem and len(self.ddpg.memory) <\
                        self.config.EXP_REPLAY_SIZE:
                    action = self.hfo_env.action_space.sample()
                else:
                    # When gen_mem is done, saves experiences and starts a new
                    # frame counting and starts the learning process
                    if self.gen_mem:
                        self.gen_mem_end(episode+1)
                    # Gets the action
                    action = self.ddpg.get_action(frame)
                    action = (action + np.random.normal(
                        0, 0.1, size=self.hfo_env.action_space.shape[0])).clip(
                        self.hfo_env.action_space.low,
                        self.hfo_env.action_space.high)
                    action = action.astype(np.float32)
                    step += 1

                if action < -0.68:
                    self.taken_action[0] += 1
                elif action < 0.36:
                    self.taken_action[1] += 1
                else:
                    self.taken_action[2] += 1

                # Calculates results from environment
                next_state, reward, done, status = self.hfo_env.step(
                    action)

                if not done:
                    next_frame = self.ddpg.stack_frames(next_state, done)
                    intermed_buffer.append((frame, action,
                                            reward, next_frame, int(done)))
                else:
                    if status == hfo.GOAL:
                        bad_buffer.append(frame)
                    else:
                        good_buffer.append(frame)
                # Calculate intermadiate rewards when the episode is finished
                if not self.gen_mem and not self.test:
                    self.ddpg.update()

                if done:
                    next_state = np.zeros(state.shape)
                    next_frame = np.zeros(frame.shape)
                    ibuffer = self.calc_rewards(good_buffer,
                                                bad_buffer,
                                                intermed_buffer,
                                                status != hfo.GOAL)
                    ibuffer = ibuffer + \
                        [(frame, action, reward, next_frame, int(done))]
                    epi_rewards = np.sum([x[2] for x in ibuffer])
                    self.ddpg.writer.add_scalar(
                        f'epi_reward_{self.unum}_gexp', epi_rewards,
                        global_step=episode)
                    if episode % 100 == 0 and episode > 10 and self.goals > 0:
                        print('Episode reward sum:', epi_rewards)
                        print('Total goals taken:', self.goals)
                        self.goals = 0
                    print('Action balance:')
                    print('Move:',
                          self.taken_action[0],
                          '- Intercept:',
                          self.taken_action[1],
                          '- Block:',
                          self.taken_action[2])

                if status == hfo.GOAL:
                    self.goals += 1
                frame = next_frame
                state = next_state
                if done:
                    break
                self.frame_idx += 1

            for x in ibuffer:
                self.ddpg.append_to_replay(*x)
            if not self.gen_mem or self.test:
                self.save_modelmem(episode)
            for i in range(len(self.taken_action)):
                self.taken_action[i] = 0
            self.bye(status)
