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

"""
    THIS FILE CONTAINS A MODEL THAT LEARNS THE DYNAMICS OF THE SYSTEM, MEANING
    THAT GIVEN AN ACTION AND A STATE, THIS MODEL WILL PRODUCE THE NEXT STATE
    THAT THE AGENT WILL TRANSITION TO FOLLOWING A FUNCTIONAL APPRIOXIMATION
    TO THE TRUE DYNAMICS OF THE SYSTEM.  I OPTED TO SPLIT UP THE MODEL THAT
    PRODUCES THE NEXT STATE AND REWARD SO THAT I COULD PERFORM GUIDED LEARNING.
    THE FILE IS SPLIT UP INTO TWO MODELS BASED UPON WHETHER THE STATE SPACE IS
    CONTINUOUS OR DISCRETE.
"""

class NN_TRANSITION_DYNAMICS_MODEL_DISC(torch.nn.Module):
    """
    MODEL OF ENVORMENT CLASS: THIS WILL SIMULATE SIMULATE THE ENVIRMOMENTS DYNAMICS
    GIVEN THE CURRENT STATE AND AN ACTION IT WILL PRODUCE A NEW STATE THAT THE
    AGENT WILL TRANSITION TO. THIS MODEL ASSUMES THAT THE TOTAL NUMBER OF STATES
    IS KNOW TO THE AGENT A-PRIORI, AND IS THEREFORE LIMITED IN ITS SCOPE TO
    VERY SIMPLE ENVIROMENTS.
    """
    def __init__(self, state_size, action_size, states, hidden_layer = 64):
        super(RWS_DISCRETE_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        self.actions = actions
        self.epsilon = torch.tensor(0.0000001)
        # 1 hidden layer
        self.linear1 = torch.nn.Linear(state_size + action_size, hidden_layer)
        # 2 hidden layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden layer
        self.linear4 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 4 hidden softmax layer
        self.linear5 = torch.nn.Linear(hidden_layer, states)
        # output
        self.softmax = torch.nn.Softmax(dim=0)
        # add a stacked output for vector calc
        self.outputstacked = torch.nn.Softmax(dim=1)
        # distribution variable for sampling
        self.dist = lambda prob: Categorical(prob)

    def sample_next_state(self, state, action):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor([action])])
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

    def forward(self, prev_state, prev_action, current_state):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(prev_state), torch.FloatTensor(prev_action)],1)
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
        return torch.log(torch.gather(output, 1, current_state) + self.epsilon)

class NN_TRANSITION_DYNAMICS_MODEL_CONT(torch.nn.Module):
    """
    MODEL OF ENVORMENT CLASS: THIS MODEL ASSUMES A CONTINUOUS REPRESENTATION FOR
    ITS SPACE OF POSSIBLE TRANSITIONS. IN THIS WAY WE LARGELY AVOID THE ISSUES
    OF SUPPORT PRESENTED BY DISCRETE ACTION SPACES, WHERE THE TOTAL NUMBER OF
    POSSIBLE STATES IS COUNTABLY INFINITE. HERE LETS JUST BEEF UP A NETWORK
    AND THEN PARAMETERIZE A MULTI-VARIATE NORMAL. THIS WILL WILL BE A CHEESE
    VERSION WHERE WE WILL PARAMETERIZE A CHOLESKY DECOMP AND THEN ADD EPSILON
    TO GUARANTEE NON-NEGATIVE DEFINITENESS.
    """
    def __init__(self, state_size, action_size, hidden_layer = 128):
        super(RWS_CONT_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.action_size = action_size
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(state_size + action_size, hidden_layer)
        # 2 hidden relu layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden softmax layer
        self.linear3 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden softmax layer
        self.linear4 = torch.nn.Linear(hidden_layer, hidden_layer)
        # output
        self.output = torch.nn.Linear(hidden_layer, state_size**2 + state_size)

    def sample_next_state(self, state, action):
        # input
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor([action])])
        # first layer
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        # second layer
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        # third layer
        probabilities = self.linear3(probabilities)
        probabilities = self.relu(probabilities)
        # fourth layer
        probabilities = self.linear4(probabilities)
        probabilities = self.relu(probabilities)
        # output to give parameters
        params = self.output(probabilities)
        # parameterize model
        mean = params[0:self.state_size]
        # parameterize cholesky fac
        L = self.upper_triag(params[self.state_size:]) + \
                        # additional cheese parameter for degeneracy
                        self.epsilon*torch.eye(self.state_size)
        # compute the actual covariance matrix used
        cov = torch.mm(L,L.t())
        # set MVT
        mvn = MultivariateNormal(mean, cov)
        # sample the next state
        next_state = mvn.sample()
        # return
        return next_state

    def forward(self, state, action, next_state):
        # input
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(action)],1)
        # first layer
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        # second layer
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        # third layer
        probabilities = self.linear3(probabilities)
        probabilities = self.tanh(probabilities)
        # output to give parameters
        parameters = self.output(probabilities)
        # use reparameterization trick to generate all the log_probs
        mean = torch.max(torch.min(torch.tensor(2.0), parameters[:,0]), -torch.tensor(2.0))
        variance =  torch.min(torch.ones(torch.mul(parameters[:,1], parameters[:,1]).size()), torch.mul(parameters[:,1], parameters[:,1])) + 0.01
        # compute log prob explicitly
        expval = next_state.reshape(-1).float() - mean
        log_prob = -1 * torch.sum(expval * expval) / (2*variance) - 0.5 * torch.log(2 * 3.141592653 * variance)
        # return it
        return log_prob

class DDP_TRANSITION_DYNAMICS_MODEL_DISC(torch.nn.Module):

    def __init__(self, state_size, action_size, states, hidden_layer = 64):
        super(RWS_DISCRETE_POLICY, self).__init__()
        self.epsilon = 0.000001

    def forward(self):
        print("wow")
        return None
