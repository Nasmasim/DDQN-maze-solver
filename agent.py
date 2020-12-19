"""
FINAL VERSION, 
Nasma Dasser
"""

import numpy as np
import collections
import torch

class Agent: 
    def __init__(self): 
    
        # ---------------------
        # INITIAL MODEL PARAMETERS
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None    
        # Define the discrete actions 
        self.actions = [0,1,2]
        # Defien the continous actions
        self.continuous_actions = np.array([[0.02, 0],  # GO right
                                            [0, 0.02],  # GO up
                                            [0, -0.02]],# GO down
                                      dtype=np.float32)          
        # ---------------------
        # REPLAY BUFFER
        # batchsize
        self.batch_size = 256
        # initialize buffer
        self.replay_buffer = ReplayBuffer()   
        
        # ---------------------
        # GREEDY CHECKER
        self.greedy_checker = GreedyChecker()
        # latest weight update
        self.last_update = False
        # when he should evaluate greedy
        self.check_greedy_policy = False
        # check every 1500 steps the greedy policy
        self.greedy_frequency = 1500
        
        # ---------------------
        # DEEP Q-NETWORK
        # Call Network to main function
        self.dqn = DQN()
        # frequency of updating target network
        self.update_target_network_frequency = 50
        
        # ---------------------
        # REWARD BASED EPSILON DECAY
        # initial value of epsilon
        self.epsilon = 1       
        # minimum value for epsilon
        self.epsilon_min = 0.05    
        self.epsilon_max = 1.15 
        # decay of epsilon
        self.epsilon_decay = 0.15
        # delta change
        self.delta = 0.00009

        # ---------------------
        # INITIAL EPISODE PARAMETERS
        # number of steps taken in an episode
        self.num_steps_taken = 0
        # distance to the goal
        self.distance_to_goal = float('inf')
        # initial length
        self.episode_length = 500
        # count steps to goal
        self.count_steps_to_goal = 0            

        # ---------------------
        # DIFFICULT MAZE CHECKER
        # check if this is a difficult maze
        self.check_difficult_maze = False
        self.count_difficult_steps = 0
        self.set_to_difficult_mode = False
        # parameters for a difficult maze
        self.epsilon_difficult = 1
        self.epsilon_difficult_min = 0.1
        self.epsilon_difficult_decay = 0.98
        self.difficult_check_frequency = 3000
        
        # print function 
        self.printit = False
               
# -----------------------------------------------------------------------------
# HAS FINISHED EPISODE

    def has_finished_episode(self):
        # ---------------------      
        # CHECK IF THIS IS A DIFFICULT MAZE
        # check if difficult maze and increase episode length
        if 5000 < self.num_steps_taken < 10000:
            self.check_difficult_maze = True
        
        # ---------------------      
        # CHECK IF EPISODE FINISHED
        if self.num_steps_taken % self.episode_length == 0:
            # if this is a diddicult maze: exponentially decay epsilon 
            if self.set_to_difficult_mode:
                # decay epsilon exponentially
                self.decay_epsilon_exponentially()
                
                # ---------------------      
                # CHECK IF ITS TIME TO EVALUATE GREEDY POLICY
                # also check for the greedy policy in this case
                if self.num_steps_taken % self.difficult_check_frequency == 0:
                    self.check_greedy_policy = True
                    # if we are in greedy dont update the target network
                    self.update_target_network_frequency = 0
            
            # if not a difficult episode
            else:
                # decay epsilon 
                self.epsilon = max(self.epsilon_min, round(self.epsilon_max - self.epsilon_decay,2))
                self.epsilon_max = self.epsilon
                
                # ---------------------      
                # CHECK IF ITS TIME TO EVALUATE GREEDY POLICY  
                # check every 1500 episodes
                if self.num_steps_taken % self.greedy_frequency == 0:
                    # Initialize greedy policy
                    self.check_greedy_policy= True              
                    # Temporarily stop updating target Network
                    self.update_target_network_frequency = 0
                    
                # decay epsilon
                if self.num_steps_taken % (10*self.episode_length) == 0 and not self.check_greedy_policy:
                    # not greedy policy, then make epsilon bigger
                    self.epsilon_min = min(self.epsilon_min+0.05,1)
            
            # end episode
            return True
        else:
            # continue episode
            return False

    def decay_epsilon_exponentially(self):
        # exponentially decay epsilon for difficult mazes
        if self.epsilon_difficult > self.epsilon_difficult_min:
            # decay epsilon
            self.epsilon_difficult *= self.epsilon_difficult_decay
  
# -----------------------------------------------------------------------------
# HAS FINISHED EPISODE
    def get_next_action(self, state):
        
        # Initialize the next actions tep
        action = self.find_next_action(state)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return self._discrete_action_to_continuous(action)
    
# -----------------------------------------------------------------------------
# SET NEXT STATE AND DISTANCE
    def set_next_state_and_distance(self, next_state, distance_to_goal):
         
        if self.printit:
            # print for debug
            if self.num_steps_taken % 1000 ==0:
                print(self.num_steps_taken, self.episode_length, self.epsilon, self.check_difficult_maze)
            
        # --------------------- 
        # TIME TO CHECK GREEDY POLICY
        if self.check_greedy_policy:
         
            # check if we reached the goal in 1000 steps and update target network weights
            if self.greedy_checker.min_steps_to_goal == 1000: 
                # check the distance 
                self.greedy_checker.check_distance_and_update_weights(current_distance_to_goal = distance_to_goal, 
                                                                          weights = self.dqn.q_target_network.state_dict())
            # count the number of steps to the goal
            self.count_steps_to_goal +=1
          
            # check if we reach the goal during greedy episode         
            if distance_to_goal < 0.03:
                              
                if self.count_steps_to_goal < self.greedy_checker.get_min_steps_to_goal():
                    
                    # if this has improved, update the target network
                    weights = self.dqn.q_target_network.state_dict()
                    # update the weights in target network
                    self.greedy_checker.reached_goal_update_weights(count_steps_to_goal= self.count_steps_to_goal, weights = weights)
                   
                    if self.count_steps_to_goal <=100 and distance_to_goal <0.03:
                        # set the parameters to full exploitation !                     
                        self.epsilon = 0
                        self.epsilon_min = 0
                        self.epsilon_decay = 0
                        self.epsilon_max = 0
                        # dont update network anymore
                        self.update_target_network_frequency = 0
            
            # if didnt reach the goal         
            elif self.count_steps_to_goal >100 and self.epsilon !=0:
                # restart the learning process
                self.check_greedy_policy = False
                self.count_steps_to_goal = 0
                self.update_target_network_frequency = 50
                
                # check if this is a difficult maze
                if self.check_difficult_maze: 
                   # if didnt reach the goal in first 200 steps
                    if self.count_steps_to_goal > 200:                       
                        self.episode_length = 800
                        self.count_difficult_steps = 0
                        self.epsilon = self.epsilon_difficult
                        self.check_difficult_maze = False
                        #print('setting to difficult mode')
                        self.set_to_difficult_mode = True
                              
        
        self.distance_to_goal = distance_to_goal
        
        # Convert the distance to a reward
        reward = self._calculate_reward(distance_to_goal)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Add transition to the replay buffer
        self.replay_buffer.add_to_buffer(transition)
        
        # check it has enough samples     
        if self.replay_buffer.check_enough_samples(self.batch_size):
            batch = self.replay_buffer.sample_random_transitions(self.batch_size)
            # train the network
            _ = self.dqn.train_network(batch)
            
            # update epsilon 
            self.epsilon = max(self.epsilon_min, self.epsilon-self.delta)
            
            # If not in greedy mode, update target network
            if self.update_target_network_frequency !=0:
                if self.num_steps_taken % self.update_target_network_frequency ==0:
                    self.dqn.update_target_network()          
 
    # convert distance to reward
    def _calculate_reward(self, distance_to_goal):
        return (1 - (4*distance_to_goal))**3
    
    # convert discrete to continous action
    def _discrete_action_to_continuous(self, discrete_action):
        return self.continuous_actions[discrete_action[0]]
            
# -----------------------------------------------------------------------------
# FIND ACTION
    def find_next_action (self, state):
        # return greedy policy using target network
        if self.check_greedy_policy:         
            return [self.dqn.q_target_network.forward(torch.tensor(state)).argmax()]    
        #else use epsilon greedy
        elif np.random.random() >= self.epsilon:
             return [self.dqn.q_network.forward(torch.tensor(state)).argmax().item()]     
        else:
             return np.random.choice(self.actions, 1, p=[1/3, 1/3, 1/3])
    
# -----------------------------------------------------------------------------
# GET GREEDY ACTION
    def _greedy_evaluation(self):
        # weights not none
        if not self.last_update and self.greedy_checker.check_weights_not_none():
            last_weights = self.greedy_checker.get_last_weights()
            # last update for q_target network
            self.dqn.q_target_network.load_state_dict(last_weights)
            # this is the last update
            self.last_update = True
            # make sure we are not training
            self.dqn.q_target_network.eval()

 
    def get_greedy_action(self, state):
        # make sure evaluation mode
        self._greedy_evaluation()
        # greedy with target network
        q = self.dqn.q_target_network.forward(torch.tensor(state)).argmax()
        # return discrete action
        return self._discrete_action_to_continuous([q.item()])
        
# -----------------------------------------------------------------------------
# NEURAL NETWORK
class Network(torch.nn.Module):

    # The class initialisation function.
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=300)
        self.layer_2 = torch.nn.Linear(in_features=300, out_features=300)
        self.layer_3 = torch.nn.Linear(in_features=300, out_features=150)
        self.layer_4 = torch.nn.Linear(in_features=150, out_features=150)
        self.output_layer = torch.nn.Linear(in_features=150, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. 
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        output = self.output_layer(layer_4_output)
        return output

# -----------------------------------------------------------------------------
# DQN
class DQN:

    # the class initialisation function 
    def __init__(self):
        # ---------------------
        # CREATE NETWORKS
        # Using Double Q-Learning 
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Target network
        self.q_target_network = Network(input_dimension=2, output_dimension=3)
        # discount rate
        self.gamma = 0.99

        # ---------------------
        # OPTIMIZER PARAMETERS
        self.learning_rate = 0.005
        # Optimize using Q-network
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def train_network(self, batch):
        # ---------------------
        # TRAIN NETWORK
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition
        loss = self._calculate_loss(batch)
        # compute gradient based on this loss, wrt to Q-network parameters
        loss.backward()
        # take one gradient step to update Q-network
        self.optimiser.step()
        # return the loss as a scalar 
        return loss.item()
    
    def _calculate_loss(self, transition):
        # ---------------------
        # CALCULATE LOSS
        # get the tensors 
        states, actions, rewards, next_states = transition       
        # argmax q-target
        q_t_s = self.q_target_network.forward(next_states).detach().max(1)[0].unsqueeze(1)        
        q_bell = rewards + self.gamma * q_t_s        
        # q-network
        q_t = self.q_network.forward(states)  
        q_t_a = torch.gather(q_t, 1, actions)        
        return torch.nn.MSELoss()(q_t_a, q_bell)

    def update_target_network(self):
        # ---------------------
        # UPDATE TARGET NETWORK
        self.q_target_network.load_state_dict(self.q_network.state_dict())

# -----------------------------------------------------------------------------
# REPLAY BUFFER
class ReplayBuffer:

        def __init__(self):
            # initialize with maximum capciaty of 5000
            self.buffer = collections.deque(maxlen=10**6)      
        
        def add_to_buffer(self, transition):
            # class function which appends a tuple to buffer container
            self.buffer.append(transition)
        
        def size(self):
            # get the size of the buffer
            return len(self.buffer)
        
        def check_enough_samples(self,batch_size):
            # check id enough samples in the buffer
            return self.size() >= batch_size
        
        def sample_random_transitions(self, batch_size):
            # function to sample random mini batch of transitions from buffer
            buffer_size = self.size()
            # sample randomly
            minibatch_index = np.random.choice(np.arange(buffer_size),
                                               size = batch_size,
                                               replace = False)
            # Get list of transitions:
            transitions_list = [self.buffer[index] for index in minibatch_index]
            
            #Get elementwise transitions
            states = torch.from_numpy(np.vstack([col[0] for col in transitions_list if col is not None]))
            actions = torch.from_numpy(np.vstack([col[1] for col in transitions_list if col is not None]))
            rewards = torch.from_numpy(np.vstack([col[2] for col in transitions_list if col is not None]))
            next_states = torch.from_numpy(np.vstack([col[3] for col in transitions_list if col is not None]))
            
            return (states.float(), actions.long(), rewards.float(), next_states.float())

# -----------------------------------------------------------------------------
# GREEDY POLICY CHECKER
class GreedyChecker:
    def __init__(self):
        # intialize
        self.min_steps_to_goal = 1000
        self.min_distance_to_goal = 1
        self.weights = None
        
    def check_distance_and_update_weights(self, current_distance_to_goal, weights):
        # checking for the first time
        if current_distance_to_goal < self.min_distance_to_goal and self.min_steps_to_goal == 1000:
            self.min_distance_to_goal = current_distance_to_goal
            self.weights = weights
    
    def reached_goal_update_weights(self, count_steps_to_goal, weights):
        self.min_steps_to_goal = count_steps_to_goal
        self.weights = weights
            
    def get_last_weights(self):
        return self.weights  
    
    def get_min_steps_to_goal(self):
        return self.min_steps_to_goal
    
    def check_weights_not_none(self):
        return self.weights is not None











    
        
        
        
        
        
