import sys
import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch
from Part_2_policy_MLP import policy_MLP
from Part_2_baseline_MLP import baseline_MLP
import torch.optim as optim
import torch.nn as nn
import copy

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
  
  # Learner setup block
  torch.manual_seed(seed)
  ####### Start
  device = "cpu"
  my_policy = policy_MLP()
  my_baseline = baseline_MLP()
  my_policy.to(device)
  my_baseline.to(device)
  stepsizes = 0.0003
  stepsizes_discount_ratio = 1/1.003
  epoches_discount_ratio =1
  mini_batch_discount_ratio = 1.003
  rand_generator = np.random.RandomState(seed)
  batch_size = 2000
  lambda_ratio = 1
  discount_ratio = 1
  range_eps = 0.2
  num_epochs = 9
  mini_batch_size = 100 # it's actually ~500 transitions. But we cannot make mini_batch size less than 1. 
  num_mini_batch = int(batch_size/mini_batch_size)
  max_episode_length = 1000
  state_batch = torch.zeros([batch_size,max_episode_length,o_dim], dtype=torch.float32, device=device)
  action_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.int, device=device)
  reward_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
  time_steps_batch = torch.zeros([batch_size,1], dtype=torch.long, device=device)
  goal_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
  advantage_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
  prob_batch = torch.zeros([batch_size,max_episode_length], dtype=torch.float32, device=device)
  seq_state_batch = torch.zeros([batch_size,o_dim], dtype=torch.float32, device=device)
  seq_action_batch = torch.zeros([batch_size,1], dtype=torch.int, device=device)
  seq_goal_batch = torch.zeros([batch_size,1], dtype=torch.float32, device=device)
  seq_advantage_batch = torch.zeros([batch_size,1], dtype=torch.float32, device=device)
  seq_prob_batch = torch.zeros([batch_size,1], dtype=torch.float32, device=device)
  stepsizes_record = []
  num_epochs_record = []
  num_mini_batch_record = []
  index_record = []
  episode_counter = 0 # counter of number of episodes
  time_counter = 0 # number of time steps of current episodes
  transition_counter = 0
  going_to_update = False  #If we have collected enough transitions and is ready to update
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
        transition_counter +=1
    if ((transition_counter) % batch_size) ==0:
        going_to_update = True
    if going_to_update and done:
        policy_optimizer = optim.Adam(my_policy.parameters(), lr=stepsizes)
        baseline_optimizer = optim.Adam(my_baseline.parameters(), lr=stepsizes)
        #Loss function contains two terms
        # For each episode, we calculate the loss
        previous_policy_net = copy.deepcopy(my_policy)
        previous_baseline_net = copy.deepcopy(my_baseline)

        for e in range(episode_counter):
            # we first calculate the Goal (return) for every time steps
            max_time_step = time_steps_batch[e]
            goal_batch[e][max_time_step-1] = reward_batch[e][max_time_step] # set the last return to be the last reward
            for t in reversed(range(max_time_step-1)):
                # From the last time steps going backward
                next_state = state_batch[e][t+1]
                old_next_state_value = previous_baseline_net(next_state)
                temp = (reward_batch[e][t+1] + discount_ratio*((1-lambda_ratio)*old_next_state_value + lambda_ratio*goal_batch[e][t+1]))
                goal_batch[e][t] = float(temp.cpu().detach())
        # calculate advantage 
        for e in range(episode_counter):
            # we first calculate the Goal (return) for every time steps
            max_time_step = time_steps_batch[e]
            for t in reversed(range(max_time_step)):
                # From the last time steps going backward
                current_state = state_batch[e][t] # the observation at time step t
                old_current_state_value = previous_baseline_net(current_state)
                current_goal = goal_batch[e][t]
                advantage_batch[e][t] = float((current_goal- old_current_state_value).cpu().detach().numpy())

        #calculate probability
        for e in range(episode_counter):
            # we first calculate the Goal (return) for every time steps
            max_time_step = time_steps_batch[e]
            for t in reversed(range(max_time_step)):
                # From the last time steps going backward
                current_action = action_batch[e][t] # the observation at time step t
                current_state = state_batch[e][t] # the observation at time step t
                old_output = previous_policy_net(current_state)
                old_action_probability = old_output[current_action]
                prob_batch[e][t] = float(old_action_probability.cpu().detach().numpy())
        #calculate goal
        update_counter = 0
        for e in range(episode_counter):
            max_time_step = time_steps_batch[e]
            for t in reversed(range(max_time_step)):
                seq_state_batch[update_counter] = state_batch[e][t]
                seq_action_batch[update_counter] = action_batch[e][t]
                seq_goal_batch[update_counter] = goal_batch[e][t]
                seq_advantage_batch[update_counter] = advantage_batch[e][t]
                seq_prob_batch[update_counter]= prob_batch[e][t]
                update_counter +=1
                if update_counter == batch_size:
                    break
            if update_counter == batch_size:
                    break
        # Normalize the advantage batch
        adv_mean = torch.mean(seq_advantage_batch)
        adv_mean = float(adv_mean.cpu().detach().numpy())
        adv_variance = torch.var(seq_advantage_batch)
        adv_variance = float(adv_variance.cpu().detach().numpy())
        seq_advantage_batch = torch.true_divide(seq_advantage_batch - adv_mean, adv_variance)
        for epoch in range(round(num_epochs)+1):
            new_index = torch.randperm(batch_size)
            seq_state_batch=seq_state_batch[new_index]
            seq_action_batch=seq_action_batch[new_index]
            seq_goal_batch = seq_goal_batch[new_index]
            seq_advantage_batch = seq_advantage_batch[new_index]
            seq_prob_batch = seq_prob_batch[new_index]
            for m in range(num_mini_batch):
                first_term = 0
                second_term = 0
                policy_optimizer.zero_grad()
                baseline_optimizer.zero_grad()
                for transition in range(m*round(mini_batch_size),(m+1)*round(mini_batch_size)):
                # we first calculate the Goal (return) for every time steps
                    #notice now the goal here is the lambda return
                    current_state = seq_state_batch[transition] # the observation at time step t
                    current_state_value = my_baseline(current_state)
                    current_action = seq_action_batch[transition] # the action at time step t
                    current_goal = seq_goal_batch[transition] # G_T
                    current_advantage = seq_advantage_batch[transition] #H_T in the first run
                    output = my_policy(current_state) # probability of actions
                    action_probability = output[int(current_action.cpu())] #probability of current action 
                    old_action_probability = seq_prob_batch[transition]
                    p_ratio = action_probability/old_action_probability
                    first_term += min(p_ratio*current_advantage , torch.clamp(p_ratio,1-range_eps, 1+range_eps)*current_advantage)
                    second_term += torch.square(current_goal - current_state_value)
                mean_first_term = first_term/round(mini_batch_size)
                mean_second_term = second_term/round(mini_batch_size)
                loss =  -mean_first_term + mean_second_term
                loss.backward()
                policy_optimizer.step()
                baseline_optimizer.step()
        for t in range(batch_size):
            seq_state_batch[t] = 0
            seq_action_batch[t] = 0
            seq_goal_batch[t] = 0
            seq_advantage_batch[t] = 0
            seq_prob_batch[t] = 0
        for e in range(episode_counter):
            state_batch[e] = 0
            action_batch[e] = 0
            reward_batch[e] = 0
            time_steps_batch[e] = 0
            goal_batch[e] = 0
            advantage_batch[e] = 0
            prob_batch[e] = 0
        episode_counter = 0 # counter of number of episodes
        time_counter = 0 # number of time steps of current episodes
        transition_counter = 0
        stepsizes_record.append(stepsizes)
        num_epochs_record.append(round(num_epochs)+1)
        num_mini_batch_record.append(round(mini_batch_size))
        stepsizes *= stepsizes_discount_ratio
        mini_batch_size *= mini_batch_discount_ratio
        num_epochs *= epoches_discount_ratio
        num_mini_batch = int(batch_size/round(mini_batch_size))
        going_to_update = False
        index_record.append(steps)
    ####### End
    
    # Log
    ret += r
    if done:
      rets.append(ret)
      ret = 0
      o = env.reset()

    if (steps+1) % checkpoint == 0:
      avgrets.append(np.mean(rets))
      rets = []
      plt.clf()
      plt.plot(range(checkpoint, (steps+1)+checkpoint, checkpoint), avgrets)
      plt.pause(0.001)
  name = sys.argv[0].split('.')[-2].split('_')[-1]
  data = np.zeros((2, len(avgrets)))
  data[0] = range(checkpoint, num_steps+1, checkpoint)
  data[1] = avgrets
  np.savetxt(name+str(seed)+".txt", data)
  
  data = np.zeros((2, len(index_record)))
  data[0] = index_record
  data[1] = stepsizes_record
  np.savetxt("step_sizes"+".txt", data)
  
  data = np.zeros((2, len(index_record)))
  data[0] = index_record
  data[1] = num_mini_batch_record
  np.savetxt("mini_batch"+".txt", data)
  plt.show()


if __name__ == "__main__":
  main()
