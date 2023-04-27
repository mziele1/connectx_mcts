# connectx_mcts
A Monte Carlo tree search agent for Kaggle's Connect X competition.


## Usage
See examples.ipynb


## Requirements
* numpy
* numba
* kaggle-environments


## Features
* Numba acceleration
* Parallelization via multiprocessing
* Multiple playout types
* Agent in competition format


## Files
**mcts_agent.py** - Contains the code to run MCTS and play Connect X. Formatted in the way the Connect X competition expects, so this file can directly be submitted as an agent.

**lookahead_agent.py** - An agent that plays the next three moves, and selects the best move based on a heuristic. Used to evaluate the performance of  the MCTS agent. Code from Kaggle's Intro to Game AI course.

**run_game.ipynb** - A Jupyter notebook that shows how to use the MCTS agent by playing and visualizing a game of Connect X.


## Modifying the Agent
### Parallelization
To change the default parallelization behavior of `mcts_agent.mcts_agent`, simply change `mp_or_sp` in mcts_agent.py. The single process and multiprocess agents can also be called directly.

### Playout Type
There are two playout types:
* Light - Completely random moves
* Heavy - Select the first available: winning move, block an opponent's winning move, random move

Generally light playouts give the best results since the code is simpler and faster, allowing more of the game tree to be explored in the same amount of time. Switch between types by modifying `playout_type` in mcts_agent.py.

### Turn Time
Agents are given 2s per turn, and 60s of time that can be used throughout the game.

The agent needs extra time on its first turn for Numba JIT compilation. After compilation, any remaining time is used to build the tree.

Remaining extra time can be allocated to each turn to increase the number of simulations per turn.

By default, the extra time is already used by the agent.


## Issues
The agent works as-is, but there are a few things to be aware of.

The most straightforward way to jitclass the Nodes would be to use a dictionary with deferred type values. It doesn't work - see https://github.com/numba/numba/issues/8404.

Multiprocessing only works on Linux.

Numba caching may work locally, but does not currently work when executed by Kaggle in the competition.

Physical processor detection via psutil may also not work in the Kaggle environment in the competition.
