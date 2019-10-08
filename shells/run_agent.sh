#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HFO_mgm/bin/HFO --fullstate --no-logging --headless --defense-agents=7 --offense-npcs=7 --defense-npcs=1 --offense-team=helios --trials $1 &
sleep 5
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
python ./agents/agent.py &
sleep 1
python ./agents/agent.py &
sleep 1
python ./agents/agent.py &
sleep 1
python ./agents/agent.py &
sleep 1
python ./agents/agent.py &
sleep 1
python ./agents/agent.py &
sleep 1
python ./agents/agent.py &
sleep 1

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait