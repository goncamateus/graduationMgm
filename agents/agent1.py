import datetime
import gc
import itertools
import logging
import os
import pickle
import sys
import time
from copy import deepcopy
# from concurrent.futures import ProcessPoolExecutor
from functools import partial

import hfo
import numpy as np
import torch.multiprocessing as mp
from gym import spaces
from joblib import Parallel, delayed

from agents.base_agent import Agent
from agents_ddpg import DDPGAgent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.Neural_Networks.DDPG import DDPG
from graduationmgm.lib.utils import MemoryDeque

MEM_SIZE = 1e5


def run(port, team, actions, rewards, num_agent, ddpg, memory, test):
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
    goals = 0
    first = True
    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        episode_rewards = 0
        step = 0
        while status == hfo.IN_GAME:
            # Every time when game resets starts a zero frame
            state = env.get_state()
            # interceptable = state[-1]
            frame = ddpg.stack_frames(state, done)
            # If the size of experiences is under max_size*8 runs gen_mem
            if len(memory) < MEM_SIZE and not test:
                action = env.action_space.sample()
            else:
                # When gen_mem is done, saves experiences and starts a new
                # frame counting and starts the learning process
                if first:
                    first = False
                    print('started learning at episode', episode)
                # Gets the action
                action = ddpg.get_action(frame)
                action = (action + np.random.normal(0, 0.1, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                action = action.astype(np.float32)
                step += 1

            # Calculates results from environment
            next_state, reward, done, status = env.step(action)
            episode_rewards += reward

            if done:
                next_state = np.zeros(state.shape)
                next_frame = np.zeros(frame.shape)
                if episode % 100 == 0 and episode > 10 and goals > 0:
                    print(goals)
                    goals = 0
            else:
                next_frame = ddpg.stack_frames(next_state, done)

            memory.store((frame, action, reward, next_frame, int(done)))
            if status == hfo.GOAL:
                goals += 1
            if len(memory) > ddpg.batch_size and not test:
                ddpg.update(memory)
            if done:
                break
    env.act(hfo.QUIT)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    team = sys.argv[1]
    num_agents = sys.argv[2]
    if team == 'helios':
        team = 'HELIOS'
    elif team == 'helios19':
        team = 'HELIOS19'
    elif team == 'robocin':
        team = 'RoboCIn'
    agent = DDPGAgent(DDPG, False,
                      team=team, port=6000,
                      num_agents=int(num_agents),
                      num_ops=int(num_agents))
    processes = []
    if agent.gen_mem:
        memories = [MemoryDeque(MEM_SIZE)
                    for _ in range(agent.num_agents)]
    else:
        memories = agent.ddpg.memory
    for rank in range(agent.num_agents):
        p = mp.Process(target=run, args=(agent.port, agent.team, agent.actions,
                                         agent.rewards, agent.num_agents,
                                         agent.ddpg, memories[rank],
                                         agent.test))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    agent.save_model(bye=True)
    exit(1)
