
""" IMPORT PACKAGES """
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


class PG_DISCRETE_POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, action_size, actions, hidden_layer = 128):
        super(PG_DISCRETE_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(state_size, hidden_layer)
        # 2 hidden relu layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 2 hidden relu layer
        self.linear3 = torch.nn.Linear(hidden_layer, actions)
        # 3 output through softmax
        self.output = torch.nn.Softmax(dim=0)
        self.outputstacked = torch.nn.Softmax(dim=1)
        self.dist = lambda prob: Categorical(prob)

    def sample_action(self, state):
        # First Hidden Layer
        output = self.linear1(torch.FloatTensor(state))
        output = F.relu(output)
        # seconf Hidden Layer
        output = self.linear2(output)
        output = F.relu(output)
        # third Hidden Layer
        output = self.linear3(output)
        output = self.output(output)
        # parameterized distribution
        distrib = self.dist(output)
        # return action
        return distrib.sample()

    def logprob_action(self, state, action):
        # Nueral net with two hidden relu layers
        output = self.linear1(state)
        output = F.relu(output)
        # seconf Hidden Layer
        output = self.linear2(output)
        output = torch.relu(output)
        # third Hidden Layer
        output = self.linear3(output)
        # output = F.relu(output)
        # outputs a parameterization
        output = self.output(output)
        # parameterized distribution
        distrib = self.dist(output)
        # return log probability of an action
        return distrib.log_prob(action)

    def forward(self, state, action):
        # Nueral net with two hidden relu layers
        output = self.linear1(state)
        output = F.relu(output)
        # seconf Hidden Layer
        output = self.linear2(output)
        output = torch.relu(output)
        # third Hidden Layer
        output = self.linear3(output)
        # output = F.relu(output)
        # outputs a parameterization
        output = self.outputstacked(output)
        # return log probability of an action
        return torch.log(torch.gather(output, 1, action))

class PG_CONT_POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, action_size, optimality_input, hidden_layer = 128):
        super(PG_CONT_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(self.state_size, hidden_layer)
        # 2 hidden relu layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden softmax layer
        self.linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # last hidden layer
        self.tanh = torch.nn.Tanh()
        # output
        self.output = torch.nn.Linear(hidden_layer, self.action_size + 1)
        # distribution un-scaled over actions
        self.dist = lambda prob: Normal(torch.max(torch.min(torch.tensor(2.0), prob[0]), -torch.tensor(2.0)),  torch.min(torch.tensor(1.), prob[1]**2) + 0.01)

    def sample_action(self, state):
        # first layer
        probabilities = self.linear1(state)
        probabilities = F.relu(probabilities)
        # second layer
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        # third layer
        probabilities = self.linear3(probabilities)
        probabilities = self.tanh(probabilities)
        # output to give parameters
        param = self.output(probabilities)
        # sample action
        action = self.dist(param).sample()
        # clip action
        action = torch.max(torch.min(torch.tensor(2.0), action), -torch.tensor(2.0))
        # return
        return action

    def forward(self, state, action):
        # first layer
        probabilities = self.linear1(state)
        probabilities = F.relu(probabilities)
        # second layer
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        # third layer
        probabilities = self.linear3(probabilities)
        probabilities = self.tanh(probabilities)
        # output to give parameters
        parameters = self.output(probabilities)

        log_norm_constant = -0.5 * np.log(2 * np.pi)

        # use reparameterization trick to generate all the log_probs
        mean = torch.max(torch.min(torch.tensor(2.0), parameters[:,0]), -torch.tensor(2.0))
        variance =  torch.min(torch.ones(torch.mul(parameters[:,1], parameters[:,1]).size()), torch.mul(parameters[:,1], parameters[:,1])) + 0.01

        # compute log prob explicitly
        expval = action.reshape(-1).float() - mean
        log_prob = -1 * torch.sum(expval * expval) / (2*variance) - 0.5 * torch.log(2 * np.pi * variance)
        # return it
        return log_prob
