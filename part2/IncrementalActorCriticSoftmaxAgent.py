# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:38:11 2020

@author: puyua
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class IncrementalActorCriticSoftmaxAgent():
    ''' 
     In this Class, we implement an Incremental Actor Critic Agent. 
     The algorithm of this agent is the same as the one-steo actor-critic algorithm
     in the text book. 
     
     The only difference is that we are using neural net as the
     function approximator of the policy and the value funtion (which will be 
     used for both bootstraping and basline). Therefore, we don't make and update
     directly, instead, we uodate the network with the loss. 
     
     During the evaluation, the agent uses neural net to generate an preference
     vector that will be then used as an input to the softmax function to get the
     probability of different action selection. 
     
     Instead of using Adam optimizer, we are using SGD here. 
     '''
        
    def __init__(self, agent_info={}):
        """
        In this section, we initialize the agent with some given parameters

        Assume agent_info dict contains:
        {
            "policy_network_size": tensor
            "basline_network_size": tensor,
            "actor_step_size": float,
            "critic_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """
        # Now we substract the agent info parameter. 
        
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        #policy_network_size = agent_info.get("policy_network_size")
        #basline_network_size = agent_info.get("basline_network_size")

        self.actor_step_size = agent_info.get("actor_step_size")
        self.critic_step_size = agent_info.get("critic_step_size")
        self.discount_rate = agent_info.get("discount_rate")
        self.actions = list(range(agent_info.get("num_actions")))

        self.softmax_prob = None 
        self.last_action = None
        
        self.policy_net = MLP_policy()
        self.basline_net = MLP_baseline()
        self.policy_optimizer = optim.SGD(self.net.parameters(), self.learning_rate, momentum=0.9)
        self.basline_optimizer = optim.SGD(self.net.parameters(), self.learning_rate, momentum=0.9)
        self.policy_criterion = nn.CrossEntropyLoss()
        self.basline_criterion = nn.MSELoss()
        
        

    def agent_policy(self, observation):
        """ policy of the agent
        Args:
            An observation

        Returns:
            The action selected according to the policy
        """
        self.softmax_prob = self.policy_net.forward(observation)
        chosen_action = self.rand_generator.choice(self.actions, p=self.softmax_prob)
        self.last_action = chosen_action
        
        return chosen_action
        
    def agent_start(self, observation):
        """When the agent start
        Args:
            The observation.
        Returns:
            The first action the agent takes.
        """
        
        softmax_prob = self.policy_net.forward(observation)
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
        self.last_Action = chosen_action
        
        return chosen_action


    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """
        self.baseline__optimizer.zero_grad()
        current_state_value_estimation = self.basline_net.forward(observation)
        last_state_value_estimation = self.basline_net.forward(self.last_observation)
        baseline = last_state_value_estimation #We use state value as the baseline
        self.last_observation = observation
        
        # Now we start the optimization of the basline neural network
        #Here we set the loss for state value as (V_t - (r_{t+1} +V_{t+1}))^2
        last_state_value_error = self.baseline_criterion(last_state_value_estimation, reward + current_state_value_estimation)
        last_state_value_error.backward()
        self.basline_optimizer.step()

        # Npw we start with the optimization of the policy neural network
        last_preference = self.policy_net(self.last_observation)
        self.policy_optimizer.zero_grad()
        last_policy_error = -1*torch.log(last_preference[self.last_action])*(reward + current_state_value_estimation - baseline)
        last_policy_error.backward()
        self.policy_optimizer.step()
        
        softmax_prob = self.policy_net.forward(observation)
        chosen_action = self.rand_generator.choice(self.actions, p=softmax_prob)
        self.softmax_prob = softmax_prob
        
        return chosen_action
    
    def agent_end(self, reward):
        self.baseline__optimizer.zero_grad()
        last_state_value_estimation = self.basline_net.forward(self.last_observation)
        baseline = last_state_value_estimation #We use state value as the baseline
        
        # Now we start the optimization of the basline neural network
        #Here we set the loss for state value as (V_t - (r_{t+1} +V_{t+1}))^2
        last_state_value_error = self.baseline_criterion(last_state_value_estimation, reward)
        last_state_value_error.backward()
        self.basline_optimizer.step()

        # Npw we start with the optimization of the policy neural network
        last_preference = self.policy_net(self.last_observation)
        self.policy_optimizer.zero_grad()
        last_policy_error = -1*torch.log(last_preference[self.last_action])*(reward - baseline)
        last_policy_error.backward()
        self.policy_optimizer.step()
    


class MLP_policy(nn.Module):
    '''
    This class gives an Multiple Layer Percentron (MLP), 
    which is namely a nerual network.
    '''
    
    def __init__(self):
        '''
        This function initialize the nerual net, with the size of the neural net
        and the type of the activation function in the most outer layer. 

        Returns
        -------
        None.

        '''
        super(MLP_policy, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 10)

        
    def forward(self, input_value):
        '''
        Forward Pass of the neural net
        '''
        
        out_1 = F.relu(self.fc1(input_value)) # output of the activation function in the first layer
        out_2 = F.relu(self.fc2(out_1)) # output of the activation function in the second layer
        prediction = self.fc3(out_2) # output of the activation function in the third layer
        
        return F.softmax(prediction)
        
    

class MLP_baseline(nn.Module):
    '''
    This class gives an Multiple Layer Percentron (MLP), 
    which is namely a nerual network.
    '''
    
    def __init__(self, learning_rate):
        '''
        This function initialize the nerual net, with the size of the neural net
        and the type of the activation function in the most outer layer. 

        Returns
        -------
        None.

        '''
        super(MLP_policy, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 10)

        
    def forward(self, input_value):
        '''
        Forward Pass of the neural net
        '''
        out_1 = F.relu(self.fc1(input_value)) # output of the activation function in the first layer
        out_2 = F.relu(self.fc2(out_1)) # output of the activation function in the second layer
        prediction = self.fc3(out_2) # output of the activation function in the third layer
        
        return prediction

