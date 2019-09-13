import datetime
import itertools
import logging
import os
import pickle

import hfo
import numpy as np

from agents.base_agent import Agent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.utils import OUNoise

logger = logging.getLogger('Agent')


class DDPGAgent(Agent):

    def __init__(self, model, per):
        self.config_hyper(per)
        self.config_env()
        self.config_model(model)
        self.noise = OUNoise(self.hfo_env.action_space)

    def config_env(self):
        BLOCK = hfo.CATCH
        self.actions = [hfo.MOVE, hfo.GO_TO_BALL, BLOCK, hfo.DEFEND_GOAL]
        self.rewards = [0, 0, 0, 0]
        self.hfo_env = HFOEnv(self.actions, self.rewards, strict=True)
        self.test = False
        self.unum = self.hfo_env.getUnum()

    def config_model(self, model):

        self.ddpg = model(env=self.hfo_env, config=self.config,
                          static_policy=self.test)
        self.currun_rewards = list()
        self.model_paths = (f'./saved_agents/actor_{self.unum}.dump',
                            f'./saved_agents/critic_{self.unum}.dump')
        self.optim_paths = (f'./saved_agents/actor_optim_{self.unum}.dump',
                            f'./saved_agents/critic_optim_{self.unum}.dump')
        self.mem_path = f'./saved_agents/exp_replay_agent_{self.unum}.dump'

        if os.path.isfile(self.model_paths[0]) \
                and os.path.isfile(self.optim_paths[0]):
            self.ddpg.load_w(path_models=self.model_paths,
                             path_optims=self.optim_paths)
            print("Model Loaded")

        if not self.test:
            if os.path.isfile(self.mem_path):
                self.ddpg.load_replay(mem_path=self.mem_path)
                self.ddpg.learn_start = 0
                print("Memory Loaded")

    def episode_end(self, total_reward, episode, state, frame):
        # Get the total reward of the episode
        print(f'Player {self.unum} episode {episode} reward {total_reward}')
        self.ddpg.finish_nstep()

    def save_modelmem(self, episode=0, bye=False):
        if (episode % 100 == 0 and episode > 0 and not self.test) or bye:
            self.ddpg.save_w(path_models=self.model_paths,
                             path_optims=self.optim_paths)
            print("Model Saved")
        if (episode % 1000 == 0 and episode > 2) or bye:
            self.ddpg.save_replay(mem_path=self.mem_path)
            print("Memory Saved")

    def save_rewards(self):
        day = datetime.datetime.now().today().day
        hour = datetime.datetime.now().hour
        minute = datetime.datetime.now().minute
        final_str = str(day) + "-" + str(hour) + "-" + str(minute)
        with open('saved_agents/rewards_{}_{}.pickle'.format(self.unum,
                                                             final_str),
                  'wb+') as fiile:
            pickle.dump(self.currun_rewards, fiile)
            fiile.close()

    def bye(self, status):
        if status == hfo.SERVER_DOWN:
            if not self.test:
                self.save_modelmem(0, True)
            self.save_rewards()
            self.ddpg.save_losses(path=(
                f'./saved_agents/critic_losses_{self.unum}.pkl',
                f'./saved_agents/policy_losses_{self.unum}.pkl'))
            self.hfo_env.act(hfo.QUIT)
            exit()

    def run(self):
        self.frame_idx = 1
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = list()
            step = 0
            while status == hfo.IN_GAME:
                # Every time when game resets starts a zero frame
                if done:
                    state = self.hfo_env.get_state(strict=True)
                    frame = self.ddpg.stack_frames(state, done)

                # Gets the action
                action = self.ddpg.get_action(frame)
                action = self.noise.get_action(action, step)
                action = action.argmax()
                step += 1

                # Calculates results from environment
                next_state, reward, done, status = self.hfo_env.step(action,
                                                                     strict=True)
                episode_rewards.append(reward)

                if done:
                    # Resets frame_stack and states
                    total_reward = np.sum(episode_rewards)
                    self.currun_rewards.append(total_reward)
                    self.episode_end(total_reward, episode, state, frame)
                    next_state = np.zeros(state.shape)
                    next_frame = np.zeros(frame.shape)
                else:
                    next_frame = self.ddpg.stack_frames(next_state, done)

                self.ddpg.update(frame, action, reward,
                                 next_frame, int(self.frame_idx / 8),
                                 store=not done)

                frame = next_frame
                state = next_state

                self.frame_idx += 1
            self.save_modelmem(episode)
            self.bye(status)
