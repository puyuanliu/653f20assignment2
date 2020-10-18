# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:51:13 2020

@author: puyua
"""
from numpy import loadtxt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.use("TKAgg")

def plot():
    plt.clf()
    plt.figure(figsize=(10,7))
    reinforce_data = np.zeros(shape=(5,50))
    plt.ylim([0, 550])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,6):
        filename = "reinforce"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        reinforce_data[i-1] = data[1]
        legend_text = "seed = " + str(i)
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_reinforce_reward = np.average(reinforce_data,axis = 0)
    std_err_reinforce_reward = np.std(reinforce_data, axis=0)/np.sqrt(len(reinforce_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Average Reward",fontsize=20)
    #plt.title("Training Average Reward of Reinforce Method",fontsize=20)
    plt.savefig('reinforce.png')
    
    
    
    # For batch data
    plt.clf()
    plt.figure(figsize=(10,7))
    batch_data = np.zeros(shape=(5,50))
    plt.ylim([0, 550])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,6):
        filename = "batchac"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        batch_data[i-1] = data[1]
        legend_text = "seed = " + str(i)
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_batchAC_reward = np.average(batch_data,axis = 0)
    std_err_batchAC_reward = np.std(batch_data, axis=0)/np.sqrt(len(batch_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Average Reward",fontsize=20)
    #plt.title("Training Average Reward of Batch AC Method",fontsize=20)
    plt.savefig('batchAC.png')
    
    plt.clf()
    plt.figure(figsize=(10,7))
    my_data = np.zeros(shape=(5,50))
    plt.ylim([0, 550])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,6):
        filename = "ppo"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        my_data[i-1] = data[1]
        legend_text = "seed = " + str(i)                                                                                                                                                                                                                                                                     
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_ppo_reward = np.average(my_data,axis = 0)
    std_err_ppo_reward = np.std(my_data, axis=0)/np.sqrt(len(my_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Average Reward",fontsize=20)
    #plt.title("Training Average Reward of PPO Method",fontsize=20)
    plt.savefig('PPO.png')
    
    plt.clf()
    plt.figure(figsize=(10,7))
    my_data = np.zeros(shape=(5,50))
    plt.ylim([0, 550])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,6):
        filename = "my"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        my_data[i-1] = data[1]
        legend_text = "seed = " + str(i)
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_my_reward = np.average(my_data,axis = 0)
    std_err_my_reward = np.std(my_data, axis=0)/np.sqrt(len(my_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Average Reward",fontsize=20)
    #plt.title("Training Average Reward of My Method",fontsize=20)
    plt.savefig('my.png')
    
    
    
    # Now we start the plot of the average
    plt.clf()
    plt.figure(figsize=(10,7))
    plt.fill_between(checkpoints, ave_reinforce_reward - std_err_reinforce_reward, ave_reinforce_reward + std_err_reinforce_reward, alpha=0.2)
    plt.plot(checkpoints, ave_reinforce_reward, linewidth=1.0,
                          label="Reinforce")
    
    plt.fill_between(checkpoints, ave_batchAC_reward - std_err_batchAC_reward, ave_batchAC_reward + std_err_batchAC_reward, alpha=0.2)
    plt.plot(checkpoints, ave_batchAC_reward, linewidth=1.0,
                          label="BatchAC")
    
    plt.fill_between(checkpoints, ave_ppo_reward - std_err_ppo_reward, ave_ppo_reward + std_err_ppo_reward, alpha=0.2)
    plt.plot(checkpoints, ave_ppo_reward, linewidth=1.0,
                          label="PPO")
    
    plt.fill_between(checkpoints, ave_my_reward - std_err_my_reward, ave_my_reward + std_err_my_reward, alpha=0.2)
    plt.plot(checkpoints, ave_my_reward, linewidth=1.0,
                          label="My Method")
    plt.ylim([0, 550])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.xlabel("Check Points",fontsize=18)
    plt.ylabel("Average Reward",fontsize=18)
    #plt.title("Comparison of Training Average Reward between different agent",fontsize=18)
    plt.legend(loc="lower right",fontsize=14)
    plt.savefig('reward_comparison.png')
    
    
    
    #Plot Epochs data
    plt.clf()
    plt.figure(figsize=(10,7))
    data = loadtxt("mini_batch.txt", comments="#", delimiter=" ", unpack=False)
    plt.plot(data[0], data[1])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.xlabel("Check Points",fontsize=18)
    plt.ylabel("Mini-Batch Size",fontsize=18)
    #plt.title("Number of Epoches Used in Different Updates in My Method",fontsize=18)
    plt.savefig('mini_batch.png')
    
    
    #Plot step sizes
    plt.clf()
    plt.figure(figsize=(12,7))
    data = loadtxt("step_sizes.txt", comments="#", delimiter=" ", unpack=False)
    plt.plot(data[0], data[1])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.xlabel("Check Points",fontsize=18)
    plt.ylabel("Step Size",fontsize=18)
    #plt.title("Step Sizes Used in Different Updates in My Method",fontsize=18)
    plt.savefig('my_step_sizes.png')
    
plot()