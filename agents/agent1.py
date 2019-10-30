from base_agent import Agent as DQNAgent
from base_agent_ddpg import DDPGAgent
from graduationmgm.lib.Neural_Networks.DQN import DQN
from graduationmgm.lib.Neural_Networks.Dueling_DQN import DDQN
from graduationmgm.lib.Neural_Networks.DDPG import DDPG


def main():
    agent = DQNAgent(DDQN, False, port=8000)
    # agent = DDPGAgent(DDPG, False)
    try:
        agent.run()
    except:
        agent.bye()


if __name__ == "__main__":
    main()
