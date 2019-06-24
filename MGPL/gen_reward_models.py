
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

class TRANSITION_REWARD_MODEL_DISC(torch.nn.Module):
    """
    MODEL OF ENVORMENT CLASS: THIS CLASS MODELS THE DISTRIBUTION OVER FINITE
    STATE DISCRETE REWARD DISTRIBTIONS. NOTE THAT WE AGAIN HAVE TO ASSUME THAT
    WE KNOW A-PRIORI WHAT THE TOTAL NUMBER STATES ARE, AS WELL AS
    WHAT THE ASSOCIATED REWARD VALUES OF EACH OF THESE STATES ARE. THIS MEANS
    THAT WE NEED TO KNOW A LOT BEFORE HAND ABOUT THE ENVIROMENT TO MODEL IT. =(
    """
    def __init__(self, state_size, action_size, rewards, reward_vals = torch.tensor([0,1]), hidden_layer = 64):
        super(TRANSITION_REWARD_MODEL_DISC, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        self.rewards = rewards
        self.epsilon = torch.tensor(0.0000001)
        # 1 hidden layer
        self.linear1 = torch.nn.Linear(2*state_size + action_size, hidden_layer)
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

    def sample_reward(self, prev_state, prev_action, current_state):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(prev_state), torch.FloatTensor(prev_action), torch.FloatTensor([prev_action])])
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

    def forward(self, prev_state, prev_action, current_state, resulting_reward):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(prev_state), torch.FloatTensor(prev_action), torch.FloatTensor([prev_action])])
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
        return torch.log(torch.gather(output, 1, resulting_reward) + self.epsilon)

class TRANSITION_REWARD_MODEL_CONT(torch.nn.Module):
    """
    MODEL OF ENVORMENT CLASS: IN THIS CASE WE DONT NEED TO MAKE ANY ASSUMPTIONS
    ABOUT WHAT THE TRUE VALUES OF THE REWARD ARE BEFORE INTERACTING WITH THE
    ENVIROMENT. IN THIS CASE WE DONT NEED TO APPLY ANY SORT OF CLIPPING OPERATION
    AS THE STATESPCE CAN BE ASSUMED TO BE THE REAL LINE.
    """
    def __init__(self, state_size, action_size, hidden_layer = 128):
        super(TRANSITION_REWARD_MODEL_CONT, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        self.rewards = rewards
        self.epsilon = torch.tensor(0.0000001)
        # 1 hidden layer
        self.linear1 = torch.nn.Linear(2*state_size + action_size, hidden_layer)
        # 2 hidden layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear4 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 4 hidden softmax layer
        self.output = torch.nn.Linear(hidden_layer, 1)
        # distribution un-scaled over actions
        self.dist = lambda prob: Normal(prob[0], prob[1]**2 + self.epsilon)

    def sample_action(self, state, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(prev_state), torch.FloatTensor(prev_action), torch.FloatTensor([prev_action])])
        # Here is the forward pass
        parameters = self.linear1(input)
        parameters = F.relu(parameters)
        parameters = self.linear2(parameters)
        parameters = F.relu(parameters)
        parameters = self.linear3(parameters)
        parameters = F.relu(parameters)
        parameters = self.linear4(parameters)
        parameters = F.relu(parameters)
        parameters = self.output(parameters)
        # sample action
        reward = self.dist(parameters).sample()
        # return
        return action

    def forward(self, state, action, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(prev_state), torch.FloatTensor(prev_action), torch.FloatTensor([prev_action])])
        # Here is the forward pass
        parameters = self.linear1(input)
        parameters = F.relu(parameters)
        parameters = self.linear2(parameters)
        parameters = F.relu(parameters)
        parameters = self.linear3(parameters)
        parameters = F.relu(parameters)
        parameters = self.linear4(parameters)
        parameters = F.relu(parameters)
        parameters = self.output(parameters)

        # use reparameterization trick to generate all the log_probs
        mean = torch.max(torch.min(torch.tensor(2.0), parameters[:,0]), -torch.tensor(2.0))
        variance =  torch.min(torch.ones(torch.mul(parameters[:,1], parameters[:,1]).size()), torch.mul(parameters[:,1], parameters[:,1])) + 0.01

        # compute log prob explicitly
        expval = action.reshape(-1).float() - mean
        log_prob = -1 * torch.sum(expval * expval) / (2*variance) - 0.5 * torch.log(2 * 3.141592653 * variance)
        # return it
        return log_prob
