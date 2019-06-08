
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

class GAME_SAMPLES(torch.nn.Module):

    """ USED AS CONTAINER FOR TRAJECTORY BATCHES TO SAVE INITIALIZATION TIME """
    def __init__(self, state_size, action_size, task, trajectory_length, sample_size, action_transform, reward_shaping):
        # initialize enviroment
        self.env = gym.make(task)
        # intialize batches of states, actions, rewards
        self.states_batch = torch.zeros((sample_size, trajectory_length, state_size))
        self.actions_batch = torch.zeros((sample_size, trajectory_length, action_size))
        self.rewards_batch = torch.zeros((sample_size, trajectory_length, 1))
        # set trajectory_length and samples
        self.trajectory_length = trajectory_length
        self.sample_size = sample_size
        # see if we need to produce an int / float / array
        self.action_transform = action_transform
        # optim tensor
        self.optimality_tensor = None
        # set reward shaping
        if reward_shaping == None:
            self.reward_shaping = lambda r: r
        else:
            self.reward_shaping = reward_shaping

    """ THIS HANDLES EARLY TERMINATION OF GAME """
    def handle_completion(self, time, sample, policy, optim):
        # set reward to the final reward state
        self.rewards_batch[sample,time:,0] = torch.stack([self.reward_shaping(torch.tensor(0.)) for _ in range(self.trajectory_length - time)])
        # set the state as the state that was finished in
        self.states_batch[sample,time:,:] = torch.stack([self.states_batch[sample,time-1,:] for _ in range(self.trajectory_length - time)])
        # generate random actions to regularize
        self.actions_batch[sample,time:,:] = torch.stack([torch.tensor([policy(self.states_batch[sample,time-1,:], optim[sample,t])]) for t in range(self.trajectory_length - time)])

    """ PRODUCES SAMPLES FROM GAME """
    def sample_game(self, env, trained_policy, optimality_tensor):
        """ SAMPLE FROM GAME UNDER ENVIROMENT AND POLICY """
        # initialize enviroment
        current_state = self.env.reset()
        policy = lambda state, optim: trained_policy.sample_action(state, optim)
        # iterate over samples
        for sample in range(self.sample_size):
            # iterate through full trajectories
            for t in range(self.trajectory_length):
                # set the current state and action
                self.states_batch[sample,t,:] = torch.tensor(current_state)
                self.actions_batch[sample,t,:] = policy(self.states_batch[sample,t,:], optimality_tensor[sample,t])
                # update stacked
                action = self.action_transform(self.actions_batch[sample,t,:])
                # take a step in the enviroment
                current_state, reward, done, info = self.env.step(action)
                # add the reward
                self.rewards_batch[sample,t,0] = self.reward_shaping(torch.tensor(reward))
                # check done flag for
                if done:

                    # pass the enviroment on to handle_completion
                    self.handle_completion(t, sample, policy, optimality_tensor)
                    # reset enviroment
                    observation = self.env.reset()
                    break
            # reset enviroment if no end
            observation = self.env.reset()

        # return game samples
        return self.states_batch, self.actions_batch, self.rewards_batch
