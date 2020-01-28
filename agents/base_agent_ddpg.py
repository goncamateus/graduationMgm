import datetime
import gc
import itertools
import logging
import os
import pickle
import socket

import hfo
import numpy as np

from agents.base_agent import Agent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.utils import AsyncWrite, OUNoise

logger = logging.getLogger('Agent')


class DDPGAgent(Agent):

    memory_save_thread = None

    def __init__(self, model, per, team='base', port=6000):
        self.config_env(team, port)
        self.config_hyper(per)
        self.config_model(model)
        self.set_comm(messages=[self.hfo_env.observation_space,
                                self.hfo_env.action_space])
        self.goals = 0

    def config_env(self, team, port):
        BLOCK = hfo.CATCH
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL, BLOCK]
        self.rewards = [0, 0, 0]
        self.hfo_env = HFOEnv(self.actions, self.rewards,
                              strict=True, continuous=True, team=team, port=port)
        self.test = False
        self.gen_mem = True
        self.unum = self.hfo_env.getUnum()

    def load_model(self, model):
        self.ddpg = model(env=self.hfo_env, config=self.config,
                          static_policy=self.test)
        self.model_paths = (f'./saved_agents/DDPG/actor_{self.unum}.dump',
                            f'./saved_agents/DDPG/critic_{self.unum}.dump')
        self.optim_paths = (f'./saved_agents/DDPG/actor_optim_{self.unum}.dump',
                            f'./saved_agents/DDPG/critic_optim_{self.unum}.dump')
        if os.path.isfile(self.model_paths[0]) \
                and os.path.isfile(self.optim_paths[0]):
            self.ddpg.load_w(path_models=self.model_paths,
                             path_optims=self.optim_paths)
            print("Model Loaded")

    def load_memory(self):
        self.mem_path = f'./saved_agents/DDPG/exp_replay_agent_{self.unum}_ddpg.dump'
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

    def set_comm(self, messages=list()):
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 65432 + self.unum        # The port used by the server

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(pickle.dumps(len(messages)))
            for msg in messages:
                msg = pickle.dumps(msg)
                s.sendall(msg)

    def run(self):
        self.frame_idx = 1
        self.goals = 0
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = 0
            step = 0
            while status == hfo.IN_GAME:
                # Every time when game resets starts a zero frame
                if done:
                    state_ori = self.hfo_env.get_state()
                    interceptable = state_ori[-1]
                    state = state_ori
                    frame = self.ddpg.stack_frames(state, done)
                # If the size of experiences is under max_size*8 runs gen_mem
                if self.gen_mem and len(self.ddpg.memory) < self.config.EXP_REPLAY_SIZE:
                    action = self.hfo_env.action_space.sample()
                else:
                    # When gen_mem is done, saves experiences and starts a new
                    # frame counting and starts the learning process
                    if self.gen_mem:
                        self.gen_mem_end(episode)
                    # Gets the action
                    action = self.ddpg.get_action(frame)
                    action = (action + np.random.normal(0, 0.1, size=self.hfo_env.action_space.shape[0])).clip(
                        self.hfo_env.action_space.low, self.hfo_env.action_space.high)
                    action = action.astype(np.float32)
                    step += 1

                # if interceptable and self.gen_mem:
                #     action = np.array(
                #         [np.random.uniform(-0.68, 0.36)], dtype=np.float32)
                #     action = (action + np.random.normal(0, 0.1, size=self.hfo_env.action_space.shape[0])).clip(
                #         self.hfo_env.action_space.low, self.hfo_env.action_space.high)
                #     action = action.astype(np.float32)

                # Calculates results from environment
                next_state_ori, reward, done, status = self.hfo_env.step(
                    action)
                next_state = next_state_ori
                episode_rewards += reward

                if done:
                    # Resets frame_stack and states
                    # if not self.gen_mem:
                    #     self.ddpg.writer.add_scalar(
                    #         f'Rewards/epi_reward_{self.unum}', episode_rewards, global_step=episode)
                    self.currun_rewards.append(episode_rewards)
                    next_state = np.zeros(state.shape)
                    next_frame = np.zeros(frame.shape)
                    if episode % 100 == 0 and episode > 10 and self.goals > 0:
                        print(self.goals)
                        self.goals = 0
                else:
                    next_frame = self.ddpg.stack_frames(next_state, done)

                if status == hfo.GOAL:
                    self.goals += 1
                self.ddpg.append_to_replay(
                    frame, action, reward, next_frame, int(done))
                if not self.gen_mem and not self.test:
                    self.ddpg.update()
                frame = next_frame
                state = next_state
                if done:
                    break
                self.frame_idx += 1
            if not self.gen_mem or self.test:
                self.save_modelmem(episode)
            gc.collect()
            self.bye(status)
