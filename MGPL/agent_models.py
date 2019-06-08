
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

class RWS_DISCRETE_POLICY(torch.nn.Module):
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

class RWS_CONT_POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, action_size, optimality_input, hidden_layer = 128):
        super(RWS_CONT_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.optimality_input = optimality_input
        self.action_size = action_size
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(self.state_size + 1, hidden_layer)
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

    def sample_action(self, state, optim):
        # input
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor([optim])])
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
        param = self.output(probabilities)
        # sample action
        action = self.dist(param).sample()
        # clip action
        action = torch.max(torch.min(torch.tensor(2.0), action), -torch.tensor(2.0))
        # return
        return action

    def forward(self, state, action, optim):
        # input
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(optim)],1)
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

        log_norm_constant = -0.5 * np.log(2 * np.pi)

        # use reparameterization trick to generate all the log_probs
        mean = torch.max(torch.min(torch.tensor(2.0), parameters[:,0]), -torch.tensor(2.0))
        variance =  torch.min(torch.ones(torch.mul(parameters[:,1], parameters[:,1]).size()), torch.mul(parameters[:,1], parameters[:,1])) + 0.01

        # compute log prob explicitly
        expval = action.reshape(-1).float() - mean
        log_prob = -1 * torch.sum(expval * expval) / (2*variance) - 0.5 * torch.log(2 * 3.141592653 * variance)
        # return it
        return log_prob

class RWS_DISCRETE_TWISTED_POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, actions, optimality_input, hidden_layer = 128):
        super(RWS_DISCRETE_TWISTED_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.optimality_input = optimality_input
        self.actions = actions
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(state_size + 1, hidden_layer)
        # 2 hidden relu layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden softmax layer
        self.linear3 = torch.nn.Linear(hidden_layer, actions)
        # output
        self.softmax = torch.nn.Softmax(dim=0)
        # add a stacked output for vector calc
        self.outputstacked = torch.nn.Softmax(dim=1)
        # distribution variable for sampling
        self.dist = lambda prob: Categorical(prob)
        # for numerical stability
        self.epsilon = 0.00001

    def sample_action(self, state, optim):
        # Forward pass on probability of current variable
        input_1 = torch.cat([torch.FloatTensor(state), torch.FloatTensor([optim])])
        pos = self.linear1(input_1)
        pos = F.relu(pos)
        pos = self.linear2(pos)
        pos = F.relu(pos)
        pos = self.linear3(pos)
        pos_prob = self.softmax(pos)
        # now we consider the probability
        input_2 = torch.cat([torch.FloatTensor(state), torch.FloatTensor([1-optim])])
        neg = self.linear1(input_2)
        neg = F.relu(neg)
        neg = self.linear2(neg)
        neg = F.relu(neg)
        neg = self.linear3(neg)
        neg_prob = self.softmax(neg)
        # compute the custom softmax function
        weights = pos_prob*(1-neg_prob)
        new_softmax = weights / torch.sum(weights)
        # action
        action = self.dist(new_softmax).sample()
        return action

    def forward(self, state, action, optim):
        # Forward pass on probability of current variable
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(optim)],1)
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        pos_prob = self.outputstacked(probabilities)
        pos_out = torch.log(torch.gather(pos_prob, 1, action))
        # now we consider the probability
        non_op = torch.ones(optim.size())-optim
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(non_op)],1)
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        neg_prob = self.outputstacked(probabilities)
        neg_out = torch.log(torch.gather(neg_prob, 1, action) + self.epsilon)
        # return log probability of an action
        return pos_out + torch.log(1 - torch.exp(neg_out) + self.epsilon)


class SEQ_DISCRETE_POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, actions, optimality_input, hidden_layer = 128):
        super(SEQ_DISCRETE_POLICY, self).__init__()
        # initializations
        self.state_size = state_size
        self.optimality_input = optimality_input
        self.actions = actions
        # 1 hidden relu layer
        self.linear1 = torch.nn.Linear(state_size + optimality_input, hidden_layer)
        # 2 hidden relu layer
        self.linear2 = torch.nn.Linear(hidden_layer, hidden_layer)
        # 3 hidden softmax layer
        self.linear3 = torch.nn.Linear(hidden_layer, actions)
        # output
        self.softmax = torch.nn.Softmax(dim=0)
        # add a stacked output for vector calc
        self.outputstacked = torch.nn.Softmax(dim=1)
        # distribution variable for sampling
        self.dist = lambda prob: Categorical(prob)

    def sample_action(self, state, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(optim).reshape(-1)])
        # Here is the forward pass
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.softmax(probabilities)
        # action
        action = self.dist(probabilities).sample()
        return action

    def logprob_action(self, state, action, optim):
        # squash state and optimality stuff togather
        input = torch.cat([torch.FloatTensor(state), torch.FloatTensor(optim).reshape(-1)])
        # Here is the forward pass
        probabilities = self.linear1(input)
        probabilities = F.relu(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = F.relu(probabilities)
        probabilities = self.softmax(probabilities)
        # logprob
        log_prob = self.dist(probabilities).log_prob(action)
        return log_prob

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
        output = self.outputstacked(probabilities)
        # return log probability of an action
        return torch.log(torch.gather(output, 1, action))
