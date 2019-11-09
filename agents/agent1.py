import sys

from base_agent import Agent as DQNAgent
from base_agent_ddpg import DDPGAgent
from graduationmgm.lib.Neural_Networks.DDPG import DDPG
from graduationmgm.lib.Neural_Networks.DQN import DQN
from graduationmgm.lib.Neural_Networks.Dueling_DQN import DDQN


def main(team='base'):
    if team == 'helios':
        team = 'HELIOS'
    elif team == 'robocin':
        team = 'RoboCIn'
    agent = DQNAgent(DDQN, False, team=team, port=8000)
    # agent = DDPGAgent(DDPG, False)
    try:
        agent.run()
    except:
        agent.bye()


if __name__ == "__main__":
    main(team=sys.argv[1])
