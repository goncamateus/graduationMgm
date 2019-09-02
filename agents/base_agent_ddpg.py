import datetime
import itertools
import logging
import os
import pickle

import hfo
import numpy as np

from agents.base_agent import Agent
from graduationmgm.lib.utils import OUNoise

logger = logging.getLogger('Agent')


class DDPGAgent(Agent):

    def __init__(self, model, per):
        self.config_hyper(per)
        self.config_env()
        self.config_model(model)
        self.noise = OUNoise(self.hfo_env.action_space)

    def config_model(self, model):

        self.ddpg = model(env=self.hfo_env, config=self.config,
                          static_policy=self.test)
        self.currun_rewards = list()
        self.model_paths = (f'./saved_agents/actor_{self.unum}.dump',
                            f'./saved_agents/critic_{self.unum}.dump')
        self.optim_paths = ('./saved_agents/actor_optim.dump',
                            './saved_agents/critic_optim.dump')
        self.mem_path = './saved_agents/exp_replay_agent_{}.dump'.format(
            self.unum)

        if os.path.isfile(self.model_paths[0]) \
                and os.path.isfile(self.optim_paths[0]):
            self.ddpg.load_w(model_path=self.model_paths,
                             optim_path=self.optim_paths)
            logging.info("Model Loaded")

        if not self.test:
            if os.path.isfile(self.mem_path):
                self.ddpg.load_replay(mem_path=self.mem_path)
                self.ddpg.learn_start = 0
                logging.info("Memory Loaded")

    def episode_end(self, total_reward, episode, state, frame):
        # Get the total reward of the episode
        logging.info('Episode %s reward %d', episode, total_reward)
        self.ddpg.finish_nstep()

    def save_modelmem(self, episode):
        if episode % 100 == 0 and episode > 0 and not self.test:
            self.ddpg.save_w(path_model=self.model_paths,
                             path_optim=self.optim_paths)
        if episode % 1000 == 0 and episode > 2:
            self.ddpg.save_replay(mem_path=self.mem_path)
            logging.info("Memory Saved")

    def save_rewards(self):
        day = datetime.datetime.now().today().day
        hour = datetime.datetime.now().hour
        minute = datetime.datetime.now().minute
        final_str = str(day) + "-" + str(hour) + "-" + str(minute)
        with open('saved_agents/rewards_{}.pickle'.format(final_str),
                  'wb+') as fiile:
            pickle.dump(self.currun_rewards, fiile)
            fiile.close()

    def bye(self, status):
        if status == hfo.SERVER_DOWN:
            if not self.test:
                self.ddpg.save_w(path_model=self.model_paths,
                                 path_optim=self.optim_paths)
                print("Model Saved")
                self.ddpg.save_replay(mem_path=self.mem_path)
                print("Memory Saved")
            self.save_rewards()
            self.ddpg.save_losses()
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

                # Calculates epsilon on frame according to the stack index
                # and gets the action
                epsilon = self.config.epsilon_by_frame(
                    int(self.frame_idx / 8))
                action = self.ddpg.get_action(frame, epsilon)
                action = self.noise.get_action(action, step)
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
