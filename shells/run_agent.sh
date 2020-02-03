#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR
export PYTHONPATH=$PYTHONPATH:$DIR/..
HFO_mgm/bin/HFO --fullstate --no-logging --headless --defense-agents=3 --offense-npcs=3 --defense-npcs=1 --offense-team=$2 --defense-team=$3 --trials $4 &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 5
python ./agents/agent.py $3&
sleep 1
python ./agents/agent.py $3&
sleep 1
python ./agents/agent.py $3&
sleep 1
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait