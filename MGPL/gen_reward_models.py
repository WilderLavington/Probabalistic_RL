
""" IMPORT RELEVENT PACKAGES """
import gym
import torch
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import floor
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy
import time
import torch.utils.data
from collections import OrderedDict
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Normal
import os
import torch.distributed as dist

"""
    THIS FILE CONTAINS GENERATIVE MODELS FOR THE REWARDS PRODUCED BY THE
    ENVIROMENT. THIS IS NOTABLY DISTINCT TO THE DYNAMICS MODEL WHICH PRODUCES
    THE NEXT STATE GIVEN THE PREVIOUS STATE AND ACTION. IN THIS MODEL, WE TAKE
    THE PREVIOUS STATE AND ACTION, AND THE CURRENT STATE AND PRODUCE THE
    CURRENT ACTION.
"""

class TRANSITION_REWARD_MODEL_DESC(torch.nn.Module):
    """
    MODEL OF ENVORMENT CLASS: THIS WILL SIMULATE BOTH THE ENVIRMENT DYNAMICS
    AS WELL AS THE
    """
    def __init__(self, state_size, actions, optimality_input, hidden_layer = 64):
        super(RWS_DISCRETE_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.optimality_input = optimality_input
        self.actions = actions
        self.epsilon = torch.tensor(0.0000001)
        # 1 hidden layer
        self.linear1 = torch.nn.Linear(state_size + 1, hidden_layer)
        # 2 hidden layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear4 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 4 hidden softmax layer
        self.linear5 = torch.nn.Linear(hidden_layer, actions)
        # output
        self.softmax = torch.nn.Softmax(dim=0)
        # add a stacked output for vector calc
        self.outputstacked = torch.nn.Softmax(dim=1)
        # distribution variable for sampling
        self.dist = lambda prob: Categorical(prob)

    def sample_action(self, state, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor([optim])])
        # Here is the forward pass
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear4(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear5(probabilities)
        probabilities = self.softmax(probabilities)
        # action
        action = self.dist(probabilities).sample()
        return action

    def forward(self, state, action, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(optim)],1)
        # Here is the forward pass
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear4(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear5(probabilities)
        output = self.outputstacked(probabilities)
        # return log probability of an action
        return torch.log(torch.gather(output, 1, action) + self.epsilon)

class TRANSITION_REWARD_MODEL_CONT(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, actions, optimality_input, hidden_layer = 64):
        super(RWS_DISCRETE_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.optimality_input = optimality_input
        self.actions = actions
        self.epsilon = torch.tensor(0.0000001)
        # 1 hidden layer
        self.linear1 = torch.nn.Linear(state_size + 1, hidden_layer)
        # 2 hidden layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear4 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 4 hidden softmax layer
        self.linear5 = torch.nn.Linear(hidden_layer, actions)
        # output
        self.softmax = torch.nn.Softmax(dim=0)
        # add a stacked output for vector calc
        self.outputstacked = torch.nn.Softmax(dim=1)
        # distribution variable for sampling
        self.dist = lambda prob: Categorical(prob)

    def sample_action(self, state, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor([optim])])
        # Here is the forward pass
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear4(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear5(probabilities)
        probabilities = self.softmax(probabilities)
        # action
        action = self.dist(probabilities).sample()
        return action

    def forward(self, state, action, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(optim)],1)
        # Here is the forward pass
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear4(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear5(probabilities)
        output = self.outputstacked(probabilities)
        # return log probability of an action
        return torch.log(torch.gather(output, 1, action) + self.epsilon)
