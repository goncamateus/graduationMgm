import datetime
import os
import pickle
import socket
import sys
import threading

import numpy as np

from graduationmgm.lib.Neural_Networks.DDPG import DDPG
from graduationmgm.lib.utils import AsyncWrite
from gym import spaces

model_paths = (f'./saved_agents/DDPG/actor.dump',
               f'./saved_agents/DDPG/critic.dump')
optim_paths = (f'./saved_agents/DDPG/actor_optim.dump',
               f'./saved_agents/DDPG/critic_optim.dump')
mem_path = f'./saved_agents/DDPG/exp_replay_agent_ddpg.dump'
EXP_REPLAY_SIZE = 3e5


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
        def.observation_space = ObservationSpace(shape=shape)


def load_model(model, env):
    ddpg = model(env=env, config=config,
                 static_policy=test)
    if os.path.isfile(model_paths[0]) \
            and os.path.isfile(optim_paths[0]):
        ddpg.load_w(path_models=model_paths,
                    path_optims=optim_paths)
        print("Model Loaded")

    return ddpg


def load_memory(ddpg):
    if os.path.isfile(mem_path) and not test:
        ddpg.load_replay(mem_path=mem_path)
        print("Memory Loaded")


def save_model(ddpg, episode=0, bye=False):
    saved = False
    if (episode % 100 == 0 and episode > 0) or bye:
        ddpg.save_w(path_models=model_paths,
                    path_optims=optim_paths)
        print("Model Saved")
        saved = True
    return saved


def save_mem(ddpg, episode=0, bye=False):
    if (episode % 1000 == 0 and episode > 2 and not test) or bye:
        ddpg.save_replay(mem_path=mem_path)
        print('Memory saved')


def get_action(unum, env, ddpg):
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432 + unum
    action = 0
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                while True:
                    state = conn.recv(3664)
                    state = pickle.loads(state)
                    if not state:
                        break
                    done = conn.recv(3664)
                    done = pickle.loads(done)
                    frame = ddpg.stack_frames(state, done)
                    # If the size of experiences is under max_size*8 runs gen_mem
                    if gen_mem and len(ddpg.memory) < EXP_REPLAY_SIZE:
                        action = env.action_space.sample()
                    else:
                        # When gen_mem is done, saves experiences and starts a new
                        # frame counting and starts the learning process
                        if gen_mem:
                            gen_mem_end(episode)
                        # Gets the action
                        action = ddpg.get_action(frame)
                        action = (action + np.random.normal(0, 0.1, size=env.action_space.shape[0])).clip(
                            env.action_space.low, env.action_space.high)
                        action = action.astype(np.float32)
                        conn.sendall(pickle.dumps(action))
    return frame, action


def main(num_mates, num_ops):
    threads = [None for _ in list(range(2, 9))[:num_mates]]
    env = MockEnv(num_mates, num_ops)
    ddpg = load_model(DDPG, env)

    for i, unum in enumerate(list(range(2, 9))[:num_mates]):
        threads[i] = threading.Thread(target=run, args=(unum,))
        threads[i].start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main(sys[1], sys[2])
