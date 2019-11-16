screen -S ddpg -dm bash -c 'script -c "shells/run_agent.sh helios helios 100000" ddpg.txt'
screen -S ddqn -dm bash -c 'script -c "shells/run_agent1.sh helios helios 100000" ddqn.txt'
screen -S dqn -dm bash -c 'script -c "shells/run_agent2.sh helios helios 100000" dqn.txt'