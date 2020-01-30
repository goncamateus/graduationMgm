import datetime
import math
import os
import pickle
import socket
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from gym import spaces

from graduationmgm.lib.hyperparameters import Config
from graduationmgm.lib.Neural_Networks.DDPG import DDPG
from graduationmgm.lib.utils import AsyncWrite

model_paths = (f'./saved_agents/DDPG/actor.dump',
               f'./saved_agents/DDPG/critic.dump')
optim_paths = (f'./saved_agents/DDPG/actor_optim.dump',
               f'./saved_agents/DDPG/critic_optim.dump')
mem_path = f'./saved_agents/DDPG/exp_replay_agent_ddpg.dump'


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


def config_hyper():
    config = Config()
    # epsilon variables
    config.epsilon_start = 0.01
    config.epsilon_final = 0.01
    config.epsilon_decay = 30000
    config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + \
        (config.epsilon_start - config.epsilon_final) * \
        math.exp(-1. * frame_idx / config.epsilon_decay)

    # misc agent variables
    config.GAMMA = 0.95
    config.LR = 0.00025
    # memory
    config.TARGET_NET_UPDATE_FREQ = 1000
    config.EXP_REPLAY_SIZE = 3e5
    config.BATCH_SIZE = 64

    # Learning control variables
    config.LEARN_START = 300000
    config.MAX_FRAMES = 60000000
    config.UPDATE_FREQ = 1

    # Nstep controls
    config.N_STEPS = 1
    # if not test:
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # else:
    #     config.device = torch.device("cpu")
    return config


def load_model(model, env, config):
    ddpg = model(env=env, config=config,
                 static_policy=False)
    if os.path.isfile(model_paths[0]) \
            and os.path.isfile(optim_paths[0]):
        ddpg.load_w(path_models=model_paths,
                    path_optims=optim_paths)
        print("Model Loaded")

    return ddpg


def load_memory(ddpg):
    if os.path.isfile(mem_path):
        ddpg.load_replay(mem_path=mem_path)
        print("Memory Loaded")


def save_model(ddpg, episode=0, bye=False):
    saved = False
    ddpg.save_w(path_models=model_paths,
                path_optims=optim_paths)
    print("Model Saved")
    saved = True
    return saved


def save_mem(ddpg, episode=0, bye=False):
    ddpg.save_replay(mem_path=mem_path)
    print('Memory saved')


def do_job(env, ddpg, config, first, unum):
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432 + unum
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if first:
        connected = False
        while not connected:
            try:
                s.connect((HOST, PORT))
                connected = True
            except ConnectionError:
                connected = False
    else:
        time.sleep(0.01)
        s.connect((HOST, PORT))
    s.sendall(pickle.dumps(True))
    print('ready')
    s.close()

    action = 0
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT+20))
    s.listen()
    conn, addr = s.accept()
    with conn:
        while True:
            state_done = conn.recv(1024)
            if not state_done:
                break
            print('get action -- server')
            state, done = pickle.loads(state_done)
            frame = ddpg.stack_frames(state, done)
            # If the size of experiences is under max_size*8 runs gen_mem
            if len(ddpg.memory) < config.EXP_REPLAY_SIZE:
                action = env.action_space.sample()
            else:
                # When gen_mem is done, saves experiences and starts a new
                # frame counting and starts the learning process
                # Gets the action
                action = ddpg.get_action(frame)
                action = (action + np.random.normal(0, 0.1, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                action = action.astype(np.float32)
            conn.sendall(pickle.dumps(action))
    
    print('train server')
    s.listen()
    conn, addr = s.accept()
    print(addr)
    with conn:
        frame = action = reward = next_frame = done = None
        while True:
            tudo = conn.recv(4096)
            if not tudo:
                break
            state, action, reward, next_state, done = pickle.loads(tudo)
            frame = ddpg.stack_frames(state, False)
            if done:
                next_state = np.zeros(state.shape)
                next_frame = np.zeros(frame.shape)
            else:
                next_frame = ddpg.stack_frames(next_state, done)
    s.close()
    return frame, action, reward, next_frame, done


def main(num_mates, num_ops):
    num_mates = int(num_mates)
    num_ops = int(num_ops)
    unums = list(range(2, 9))[:num_mates]
    config = config_hyper()
    env = MockEnv(num_mates, num_ops)
    ddpg = load_model(DDPG, env, config)
    test = False
    count = 0
    first = True
    while True:
        # Get action part
        with ThreadPoolExecutor(max_workers=num_mates) as executor:
            do_job_part = partial(do_job, env, ddpg, config, first)
            conjunto = executor.map(do_job_part, unums)
            conjunto = list(conjunto)
            first = False
            for sarsd in conjunto:
                frame, action, reward, next_frame, done = sarsd
                ddpg.append_to_replay(
                    frame, action, reward, next_frame, int(done))
            if len(ddpg.memory) > 64 and not test:
                ddpg.update()
            if conjunto[0][-1]:
                count += 1
                if count % 10000 == 0 and count > 1:
                    save_model(ddpg, episode=count)
                if count % 10000 == 0 and count > 1:
                    save_mem(ddpg, episode=count)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
