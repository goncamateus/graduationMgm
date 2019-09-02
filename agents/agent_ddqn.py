import glob
import itertools
import logging
import math
import os
import sys
from pathlib import Path

import hfo
import numpy as np
import torch
from graduationmgm.Deep_Q_Networks.Dueling_DQN import Model as DuelingDQN
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.hyperparameters import Config
from graduationmgm.lib.utils import gen_mem_end, episode_end, save_modelmem, save_rewards


def main():
    # ------------------------ HYPER PARAMETERS --------------------------------- #
    config = Config()
    # epsilon variables
    config.epsilon_start = 1.0
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
    config.EXP_REPLAY_SIZE = 100000
    config.BATCH_SIZE = 64
    config.PRIORITY_ALPHA = 0.6
    config.PRIORITY_BETA_START = 0.4
    config.PRIORITY_BETA_FRAMES = 100000
    config.USE_PRIORITY_REPLAY = True

    # epsilon variables
    config.SIGMA_INIT = 0.5

    # Learning control variables
    config.LEARN_START = 100000
    config.MAX_FRAMES = 60000000
    config.UPDATE_FREQ = 1

    # Nstep controls
    config.N_STEPS = 1
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ ENVIROMENT --------------------------------------- #

    actions = [hfo.MOVE, hfo.GO_TO_BALL]
    rewards = [700, 1000]
    hfo_env = HFOEnv(actions, rewards, strict=True)
    test = False
    gen_mem = True
    unum = hfo_env.getUnum()

    logging.basicConfig(level=logging.INFO)

# ------------------------ MODEL CONFIG ------------------------------------- #

    ddqn = DuelingDQN(env=hfo_env, config=config, static_policy=test)
    currun_rewards = list()

    model_path = './saved_agents/model_{}.dump'.format(unum)
    optim_path = './saved_agents/optim_{}.dump'.format(unum)
    mem_path = './saved_agents/exp_replay_agent_{}.dump'.format(unum)

    if os.path.isfile(model_path) and os.path.isfile(optim_path):
        ddqn.load_w(model_path=model_path, optim_path=optim_path)
        logging.info("Model Loaded")

    if not test:
        if os.path.isfile(mem_path):
            ddqn.load_replay(mem_path=mem_path)
            ddqn.learn_start = 0
            logging.info("Memory Loaded")

# ------------------------ TRAINING/TEST LOOP ------------------------------- #
    frame_idx = 1
    for episode in itertools.count():
        status = hfo.IN_GAME
        done = True
        episode_rewards = list()
        while status == hfo.IN_GAME:
            # Every time when game resets starts a zero frame
            if done:
                state = hfo_env.get_state(strict=True)
                frame = ddqn.stack_frames(state, done)
            # If the size of experiences is under max_size*16 runs gen_mem
            # Biasing the agent for the Agent2d Helios_base
            if gen_mem and frame_idx / 16 < config.EXP_REPLAY_SIZE:
                action = 1 if state[-2] else 0
            else:
                # When gen_mem is done, saves experiences and starts a new
                # frame counting and starts the learning process
                if gen_mem:
                    gen_mem_end(gen_mem, episode, ddqn, frame_idx)

                # Calculates epsilon on frame according to the stack index
                # and gets the action
                epsilon = config.epsilon_by_frame(
                    int(frame_idx / 16))
                action = ddqn.get_action(frame, epsilon)

            # Calculates results from environment
            next_state, reward, done, status = hfo_env.step(action,
                                                            strict=True)
            episode_rewards.append(reward)

            if done:
                # Resets frame_stack and states
                total_reward = np.sum(episode_rewards)
                currun_rewards.append(total_reward)
                episode_end(total_reward, episode, ddqn)
                next_state = np.zeros(state.shape)
                next_frame = np.zeros(frame.shape)
            else:
                next_frame = ddqn.stack_frames(next_state, done)

            ddqn.update(frame, action, reward,
                        next_frame, int(frame_idx / 16))
            frame = next_frame
            state = next_state

            frame_idx += 1
        save_modelmem(episode, test, ddqn, model_path, optim_path,
                        mem_path, frame_idx, ddqn.experience_replay_size)
# ------------------------ QUIT --------------------------------------------- #
        if status == hfo.SERVER_DOWN:
            if not test:
                ddqn.save_w(path_model=model_path, path_optim=optim_path)
                print("Model Saved")
                ddqn.save_replay(mem_path=mem_path)
                print("Memory Saved")
            save_rewards(currun_rewards)
            hfo_env.act(hfo.QUIT)
            exit()


if __name__ == "__main__":
    main()
