# DDQN Random Maze Solver 

## Description 
Deep DQN implementation to solve a randomly generated maze problem, where an agent has to reach a given goal position. The agent follows a pre-defined set of rules: 

* The magnitude of the action vector is limited to 0.02
* No tabular reinforcement learning
* Not allowed any form of memory, other than an experience replay buffer 


<p align="center">
<img src= https://github.com/Nasmasim/DDQN-maze-solver/blob/main/results/maze_gif.gif width="30%">
</p>


## Implementation 
The general architecture of the algorithm is a Double Q-Learning algorithm with Experience Replay Buffer, and target network used to find the action with the highest Q-value, while the Q network is used to find the Q-value of this action. The implementation is split into 5 classes: 
| class | description |
| ----- | ------      |
| Agent | Agent has a set of 3 discrete actions (move up, down and right) and checks if we are in a difficult maze. If yes, then epsilon is decayed more slowly to allow more exploration. The distance between the agent and the goal is converted to a reward |
| Network | Defines a 4 hidden neural network activated by ReLu with 300 neurons each. 
| ReplayBuffer | Initialised with a large collection deque, from which we sample random transitions during training |
| GreedyChecker | Stores the minimum distance to the goal so far reached during greedy policy and if this is a good distance to the goal, update the target network weights to control updates to the Q-network and stabilise learning |

## Requirements  

``` python3, Numpy, Torch, OpenCV```

## Installation 
```
conda create -n dqn
source activate dqn

conda install python
conda install numpy
pip install torch
pip install opencv-python
```

## Running the model   
```python train_and_test.py --visualize``

