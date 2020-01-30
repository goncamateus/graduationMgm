import datetime
import itertools
import logging
import os
import pickle
import socket
import time

import hfo
import numpy as np

from agents.base_agent import Agent
from graduationmgm.lib.hfo_env import HFOEnv
from graduationmgm.lib.utils import AsyncWrite, OUNoise

logger = logging.getLogger('Agent')


class DDPGAgent(Agent):

    memory_save_thread = None

    def __init__(self, team='base', port=6000):
        self.config_env(team, port)
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

    def set_comm(self, message, port, recmsg=-1):
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = port  # The port used by the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            bom = False
            while not bom:
                try:
                    s.connect((HOST, PORT))
                    bom = True
                except ConnectionError:
                    time.sleep(0.01)
            msg = pickle.dumps(message)
            s.sendall(msg)

            if recmsg > 0:
                recv = s.recv(recmsg)
                recv = pickle.loads(recv)
                return recv

    def recv_ready(self):
        HOST = '127.0.0.1'  # The server's hostname or IP address
        PORT = 65432 + self.unum  # The port used by the server

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            print('espera ready')
            conn, addr = s.accept()
            first = False
            with conn:
                while not first:
                    if first:
                        break
                    ready = conn.recv(1024)
                    if ready:   
                        first = pickle.loads(ready)

    def get_action(self, state, done):
        PORT = 65452 + self.unum  # The port used by the server
        self.recv_ready()
        print('get_action')
        action = self.set_comm(message=(state, done), port=PORT, recmsg=1024)
        return action

    def train(self, state, action, reward, next_state, done):
        PORT = 65452 + self.unum  # The port used by the server
        self.set_comm(message=(state, action, reward, next_state, done), port=PORT)
        print('train agent')

    def run(self):
        self.goals = 0
        for episode in itertools.count():
            status = hfo.IN_GAME
            done = True
            episode_rewards = 0
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
                    if episode % 100 == 0 and episode > 1:
                        print(self.goals)
                        self.goals = 0
                    break
                state = next_state

            self.bye(status)
