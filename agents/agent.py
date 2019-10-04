from base_agent import Agent as DQNAgent
from base_agent_ddpg import DDPGAgent
from graduationmgm.lib.Neural_Networks.DQN import Model as DQN
from graduationmgm.lib.Neural_Networks.Dueling_DQN import Model as DuelingDQN
from graduationmgm.lib.Neural_Networks.DDPG import Model as DDPG


def main():
    # agent = DQNAgent(DuelingDQN, False)
    agent = DDPGAgent(DDPG, False)
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.bye()


if __name__ == "__main__":
    main()
