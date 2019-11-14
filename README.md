# Comparing DQN, Dueling Double DQN and Deep Deterministic Policy Gradient applied to Robocup Soccer Simulation 2D

This work is designed to help RoboCIn team.
Inside you'll find codes comparing each technique.

For 3000 test episodes:

 - Helios2013 vs Helios2013 -> 77,5% defenses of Helios2013
 - Helios2013 vs RoboCIn2019 -> 71% defenses of RoboCIn2019
 - RoboCIn2019 vs Helios2013 -> 77.4% defenses of Helios2013
 - RoboCIn2019 vs RoboCIn2019 -> 53.3% defenses of RoboCIn2019

100k training dqn:
 - With Helios2013 goalie:
    - 52.2% defenses against Helios2013
    - 52.2% defenses against RoboCIn2019

100k training ddqn:
 - With Helios2013 goalie:
    - 55% defenses against Helios2013
    - 55% defenses against RoboCIn2019

100k training ddpg:
 - With Helios2013 goalie:
    - 30.2% defenses against Helios2013
    - 30.2% defenses against RoboCIn2019

