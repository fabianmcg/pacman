# DQN Pacman
Author: Fabian Mora

An implementation of 4 multi agent learning algorithms for playing the Pacman game in the [Pacman Berkeley Simulator](http://ai.berkeley.edu/project_overview.html).

## Relevant implemented files
- agent.py defines the interface for all implemented Pacman agents.
- agentUtil.py contains functions for computing state representations of the game.
- rewards.py contains the internal reward function used by the agents.

Implemented agents:
- phcAgents.py based on the PHC algorithm.
- wphcAgents.py based on the WPHC algorithm.
- dqnAgents.py based on DeepQ networks.
- qdqnAgents.py based on DeepQ networks and a WPHC agent.

## Running the agents
The recommend mechanism for running an agent is to use the `run.py` script. For a list of available options and their description use:

```
python3 run.py -h        # For general options 
python3 run.py PHC -h    # For PHC options
python3 run.py WPHC -h   # For WPHC options
python3 run.py DQN -h    # For DQN options
python3 run.py WDQN -h   # For WDQN options
```
### Examples
```
python3 run.py -g 10 -t 2 WPHC           # Run the WPHC agent for 10 evaluation games and 2 training games.
python3 run.py -g 10 -t 100 DQN -L 0.001 # Run the DQN agent for 10 evaluation games and 100 training games using an optimizer learning rate of 0.001.
```

