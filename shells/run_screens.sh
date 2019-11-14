screen -S ddpg -dm bash -c 'script -c "shells/run_agent.sh robocin helios 3000" ddpg.txt'
screen -S ddqn -dm bash -c 'script -c "shells/run_agent1.sh robocin helios 3000" ddqn.txt'
screen -S dqn -dm bash -c 'script -c "shells/run_agent2.sh robocin helios 3000" dqn.txt'