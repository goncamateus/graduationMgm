import sys

from base_agent import Agent as DQNAgent
from base_agent_ddpg import DDPGAgent
from graduationmgm.lib.Neural_Networks.DDPG import DDPG
from graduationmgm.lib.Neural_Networks.DQN import DQN
from graduationmgm.lib.Neural_Networks.Dueling_DQN import DDQN


def main(team='base'):
    if team == 'helios':
        team = 'HELIOS'
    elif team == 'helios19':
        team = 'HELIOS19'
    elif team == 'robocin':
        team = 'RoboCIn'
    agent = DDPGAgent(DDPG, True, team=team, port=6000)
    try:
        agent.run()
    except:
        exit(1)


if __name__ == "__main__":
    main(team=sys.argv[1])
