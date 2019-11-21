screen -S hh -dm bash -c 'script -c "shells/run_agent1.sh helios helios 3000" heliosxhelios.txt'
screen -S rh -dm bash -c 'script -c "shells/run_agent2.sh helios helios 3000" robocinxhelios.txt'
screen -S hr -dm bash -c 'script -c "shells/run_agent3.sh helios helios 3000" heliosxrobocin.txt'
screen -S rr -dm bash -c 'script -c "shells/run_agent.sh helios helios 3000" robocinxrobocin.txt'
