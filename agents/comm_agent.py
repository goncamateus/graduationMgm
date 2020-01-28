import datetime
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

    def set_comm(self, messages=list(), recmsg=-1):
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 65432 + self.unum  # The port used by the server

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            for msg in messages:
                msg = pickle.dumps(msg)
                s.sendall(msg)
            if recmsg > 0:
                recv = s.recv(recmsg)
                return pickle.loads(recv)

    def get_action(self, state, done):
        action = set_comm(messages=[state, done], recmsg=12)
        return action

    def train(self, state, action, reward, next_state, done):
        set_comm(messages=[state, action, reward, next_state, done])

    def run(self):
        self.goals = 0
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = 0
            step = 0
            while status == hfo.IN_GAME:
                # Every time when game resets starts a zero frame
                if done:
                    state = self.hfo_env.get_state()
                action = self.get_action(state, done)
                # Calculates results from environment
                next_state, reward, done, status = self.hfo_env.step(
                    action)
                self.train(state, action, reward, next_state, done)
                episode_rewards += reward
                if status == hfo.GOAL:
                    self.goals += 1
                if done:
                    break
                state = next_state
            self.bye(status)
