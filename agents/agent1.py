import sys

from agents_ddpg import DDPGAgent
from graduationmgm.lib.Neural_Networks.DDPG import DDPG


def main(team='base', num_agents=1):
    if team == 'helios':
        team = 'HELIOS'
    elif team == 'helios19':
        team = 'HELIOS19'
    elif team == 'robocin':
        team = 'RoboCIn'
    DDPGAgent(DDPG, False,
              team=team, port=6000,
              num_agents=int(num_agents))


if __name__ == "__main__":
    main(team=sys.argv[1], num_agents=sys.argv[2])
