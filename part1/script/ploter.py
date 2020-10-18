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
    incremental_data = np.zeros(shape=(10,60))
    plt.ylim([0, 100])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,11):
        filename = "incremental"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        incremental_data[i-1] = data[1]
        legend_text = "seed = " + str(i)
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_incremental_accuracy = np.average(incremental_data,axis = 0)
    std_err_incremental_accuracy = np.std(incremental_data, axis=0)/np.sqrt(len(incremental_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Accuracies",fontsize=20)
    plt.savefig('incremental.png')
    
    
    
    # For batch data
    plt.clf()
    plt.figure(figsize=(10,7))
    batch_data = np.zeros(shape=(10,60))
    plt.ylim([0, 100])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,11):
        filename = "batch"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        batch_data[i-1] = data[1]
        legend_text = "seed = " + str(i)
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_batch_accuracy = np.average(batch_data,axis = 0)
    std_err_batch_accuracy = np.std(batch_data, axis=0)/np.sqrt(len(batch_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Accuracies",fontsize=20)
    plt.savefig('batch.png')
    
    plt.clf()
    plt.figure(figsize=(10,7))
    my_data = np.zeros(shape=(10,60))
    plt.ylim([0, 100])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    for i in range(1,11):
        filename = "my"+str(i) + ".txt"
        data = loadtxt(filename, comments="#", delimiter=" ", unpack=False) # read data from txt file
        my_data[i-1] = data[1]
        legend_text = "seed = " + str(i)
        plt.plot(data[0], data[1], label = legend_text)
        checkpoints = data[0]
    ave_my_accuracy = np.average(my_data,axis = 0)
    std_err_my_accuracy = np.std(my_data, axis=0)/np.sqrt(len(my_data))
    plt.legend(loc="lower right",fontsize=20)
    plt.xlabel("Check Points",fontsize=20)
    plt.ylabel("Accuracies",fontsize=20)
    plt.savefig('my.png')
    
    
    
    # Now we start the plot of the average
    plt.clf()
    plt.figure(figsize=(10,7))
    plt.fill_between(checkpoints, ave_incremental_accuracy - std_err_incremental_accuracy, ave_incremental_accuracy + std_err_incremental_accuracy, alpha=0.2)
    plt.plot(checkpoints, ave_incremental_accuracy, linewidth=1.0,
                          label="Incremental Method")
    
    plt.fill_between(checkpoints, ave_batch_accuracy - std_err_batch_accuracy, ave_batch_accuracy + std_err_batch_accuracy, alpha=0.2)
    plt.plot(checkpoints, ave_batch_accuracy, linewidth=1.0,
                          label="Batch Method")
    
    plt.fill_between(checkpoints, ave_my_accuracy - std_err_my_accuracy, ave_my_accuracy + std_err_my_accuracy, alpha=0.2)
    plt.plot(checkpoints, ave_my_accuracy, linewidth=1.0,
                          label="My Method")
    print(np.linalg.norm(ave_batch_accuracy - ave_my_accuracy))
    plt.ylim([0, 100])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.xlabel("Check Points",fontsize=18)
    plt.ylabel("Accuracies",fontsize=18)
    plt.legend(loc="lower right",fontsize=14)
    plt.savefig('accuracy_comparison.png')
    
    
    
    #Plot Epochs data
    plt.clf()
    plt.figure(figsize=(10,7))
    data = loadtxt("my_epochs.txt", comments="#", delimiter=" ", unpack=False)
    plt.plot(data[0], data[1])
    plt.ylim([0, 10])
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.xlabel("Check Points",fontsize=18)
    plt.ylabel("Epoches",fontsize=18)
    plt.legend(loc="lower right",fontsize=14)
    plt.savefig('my_epochs.png')
    
plot()