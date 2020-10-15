# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:37:50 2020

@author: puyua
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class policy_MLP(nn.Module):
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
        super(policy_MLP, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self, x):
        '''
        Forward Pass of the neural net
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=0)
    
        