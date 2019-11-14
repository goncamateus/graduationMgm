screen -S ddpg -dm bash -c 'script -c "shells/run_agent.sh $1 $2 3000" ddpg.txt'
screen -S ddqn -dm bash -c 'script -c "shells/run_agent1.sh $1 $2 3000" ddqn.txt'
screen -S dqn -dm bash -c 'script -c "shells/run_agent2.sh $1 $2 3000" dqn.txt'