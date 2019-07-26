from base_agent import Agent
from Deep_Q_Networks.DQN import Model as DQN
from Deep_Q_Networks.Dueling_DQN import Model as DuelingDQN


def main():
    agent = Agent(DQN, False)
    agent.run()


if __name__ == "__main__":
    main()
