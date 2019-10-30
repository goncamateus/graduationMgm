from base_agent import Agent as DQNAgent
from base_agent_ddpg import DDPGAgent
from graduationmgm.lib.Neural_Networks.DDPG import DDPG
from graduationmgm.lib.Neural_Networks.DQN import DQN
from graduationmgm.lib.Neural_Networks.Dueling_DQN import DDQN


def main():
    # agent = DQNAgent(DQN, False)
    agent = DDPGAgent(DDPG, False)
    try:
        agent.run()
    except:
        agent.bye()


if __name__ == "__main__":
    main()
