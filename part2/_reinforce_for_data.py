# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:46:43 2020

@author: puyua
"""
import sys
import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch
from Part_2_policy_MLP import policy_MLP
import torch.optim as optim
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
  seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

  # Task setup block starts
  # Do not change
  env = gym.make('CartPole-v1')
  env.seed(seed)
  o_dim = env.observation_space.shape[0]
  a_dim = env.action_space.n
  # Task setup block end
  for seed in range(1,6):
      # Learner setup block
      torch.manual_seed(seed)
      print(seed)
      ####### Start
      my_policy = policy_MLP()
      my_policy.to(device)
      optimizer = optim.Adam(my_policy.parameters(), lr=0.01)
      rand_generator = np.random.RandomState(seed)
      batch_size = 30
      # num_epochs = 10
      # mini_batch_size = 20
      # num_mini_batch = int(batch_size/mini_batch_size)
      max_episode_length = 1000
      state_batch = torch.zeros([batch_size,max_episode_length,o_dim], dtype=torch.float32, device=device)
      action_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.long, device=device)
      reward_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
      goal_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
      time_steps_batch = torch.zeros([batch_size,1], dtype=torch.long, device=device)
      episode_counter = 0 # counter of number of episodes
      time_counter = 0 # number of time steps of current episodes
      ####### End
    
      # Experiment block starts
      ret = 0
      rets = []
      avgrets = []
      o = env.reset()
      num_steps = 500000
      checkpoint = 10000
      for steps in range(num_steps):
    
        # Select an action
        ####### Start
        # Replace the following statement with your own code for
        # selecting an action
        new_o = torch.from_numpy(o)
        o = new_o.float()
        o = o.to(device)
        softmax_prob = my_policy.forward(o)
        softmax_prob = softmax_prob.cpu().detach().numpy() 
        a = rand_generator.choice([0,a_dim-1], p=softmax_prob)
        ####### End
    
        # Observe
        op, r, done, infos = env.step(a)
    
        # Learn
        ####### Start
        state_batch[episode_counter][time_counter] = o  # We previously (at time t) observed o
        action_batch[episode_counter][time_counter] = a  # We previously (at time t) observed a
        reward_batch[episode_counter][time_counter+1] = r # Now (at time t+1) we observe reward r
        o = op #update the observation
        if done:
            time_steps_batch[episode_counter] = time_counter
            time_counter = 0
            episode_counter +=1
        else:
            time_counter+=1
        if episode_counter == batch_size:
            loss = 0
            optimizer.zero_grad()
            # For each episode, we calculate the loss
            for e in range(batch_size):
                # we first calculate the Goal (return) for every time steps
                max_time_step = time_steps_batch[e]
                goal_batch[e][max_time_step-1] = reward_batch[e][max_time_step] # set the last return to be the last reward
                for t in reversed(range(max_time_step-1)):
                    # From the last time steps going backward
                    goal_batch[e][t] = reward_batch[e][t+1] + goal_batch[e][t+1]
                #Now we calculate the loss at every time step
                for t in range(max_time_step):
                    current_state = state_batch[e][t] # the observation at time step t
                    current_action = action_batch[e][t] # the action at time step t
                    output = my_policy(current_state)
                    current_goal = goal_batch[e][t]
                    current_goal = current_goal.cpu().detach().numpy()
                    action_probability = output[current_action]
                    loss += -1*torch.log(action_probability)*float(current_goal)
            loss = loss/sum(time_steps_batch)
            loss.backward()
            optimizer.step()
            state_batch = torch.zeros([batch_size,max_episode_length,o_dim], dtype=torch.float32, device=device)
            action_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.long, device=device)
            reward_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
            goal_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
            episode_counter = 0 # counter of number of episodes
        ####### End
        
        # Log
        ret += r
        if done:
          rets.append(ret)
          ret = 0
          o = env.reset()
    
        if (steps+1) % checkpoint == 0:
          avgrets.append(np.mean(rets))
          print(np.mean(rets))
          rets = []
          #plt.clf()
          #plt.plot(range(checkpoint, (steps+1)+checkpoint, checkpoint), avgrets)
          #plt.pause(0.001)
      name = sys.argv[0].split('.')[-2].split('_')[-1]
      data = np.zeros((2, len(avgrets)))
      data[0] = range(checkpoint, num_steps+1, checkpoint)
      data[1] = avgrets
      np.savetxt("reinforce"+str(seed)+".txt", data)
      #plt.show()


if __name__ == "__main__":
  main()
