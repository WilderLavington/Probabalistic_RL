
""" IMPORT PACKAGES """
import gym
import torch
from torch.distributions import MultivariateNormal, Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
from numpy import floor
import torch.multiprocessing as mp
from copy import deepcopy
import time
import torch.utils.data
from collections import OrderedDict
from torch.distributions.bernoulli import Bernoulli
import os
import torch.distributed as dist

class PG_LOSS(torch.nn.Module):
    """ POLICY GRADIENTS LOSS FUNCTION """
    def __init__(self, trajectory_length, simulations, max_ent = 0, TRPO = 0, PPO = 0, beta = None):
        """ INITIALIZATIONS """
        super(PG_LOSS, self).__init__()
        # initialize basic parameters
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.max_ent = max_ent

    def forward(self, policy, state_tensor, action_tensor, reward_tensor, \
                        cumulative_rollout):
        """ CONVERT FORMAT """
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)
        flat_cumsum = torch.flatten(cumulative_rollout, start_dim=0,end_dim=1)
        """ CALCULATE LIKLIHOOD """
        logliklihood_tensor = policy(flat_states,flat_actions)
        """ CALCULATE ADVANTAGE (MC) """
        if self.max_ent == 1:
            A_hat = -flat_cumsum.detach() - logliklihood_tensor.detach()
        else:
            A_hat = -flat_cumsum.detach()
        """ CALCULATE POLICY GRADIENT OBJECTIVE """
        expectation = torch.dot(A_hat.reshape(-1),logliklihood_tensor.reshape(-1))/self.trajectory_length
        """ RETURN """
        return expectation/self.simulations

class TRPO_LOSS(torch.nn.Module):

    """ TRUST REGION POLICY OPTIMIZATION LOSS FUNCTION """
    def __init__(self, trajectory_length, simulations, max_ent = 0, TRPO = 0, PPO = 0, beta = None):
        """ INITIALIZATIONS """
        super(TRPO_LOSS, self).__init__()
        # initialize basic parameters
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.max_ent = max_ent

    def KL_div(self, flat_states, flat_actions, policy, old_policy):
        p = torch.exp(old_policy(flat_states,flat_actions))
        log_p = old_policy(flat_states,flat_actions)
        log_q = policy(flat_states,flat_actions)
        return torch.sum(p * (log_p - log_q))

    def forward(self, policy, state_tensor, action_tensor, reward_tensor, \
                        cumulative_rollout):
        return None

class PPO_LOSS(torch.nn.Module):

    """ TRUST REGION POLICY OPTIMIZATION LOSS FUNCTION """
    def __init__(self, trajectory_length, simulations, max_ent = 0, TRPO = 0, PPO = 0, beta = None):
        """ INITIALIZATIONS """
        super(PPO_LOSS, self).__init__()
        # initialize basic parameters
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.max_ent = max_ent

    def KL_div(self, flat_states, flat_actions, policy, old_policy):
        p = torch.exp(old_policy(flat_states,flat_actions))
        log_p = old_policy(flat_states,flat_actions)
        log_q = policy(flat_states,flat_actions)
        return torch.sum(p * (log_p - log_q))

    def forward(self, policy, state_tensor, action_tensor, reward_tensor, \
                        cumulative_rollout):
        return None
