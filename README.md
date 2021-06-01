# DDQN Random Maze Solver 

## Description 
In Reinforcement Learning, an Agent interacts with an environment while evaluating its state, actions and receiving rewards. The aim is to learn which actions the agent should take to reach the goal. This project presents a deep DQN implementation to solve a randomly generated maze problem, where an agent has to reach a given goal position. The agent follows a pre-defined set of rules: 

* Training time is limited to 10 minutes, thereafter another random maze is generated
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
| Network | Defines a 4 hidden neural network activated by ReLu with 300 neurons each |
| DQN | Defines the discount rate to 0.99 and learning rate to 0.005 with Adam optimiser. We compute the gradient step to update the Q-network and for each transition compute the argmax value of the Q-value from the target network for the next state. We then compute the Bellman equation and compute the loss between the Q-value for the current state of the Q-network and the predicted one | 
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
```python train_and_test.py --visualize```

