
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
from torch.distributions.bernoulli import Bernoulli as bern

""" IMPORT PROGRAMS """
from game_enviroments import *
from agents import *
from objective_function import *
from rl_variable_imports import *

class TRAIN_AGENT(torch.nn.Module):

    """ GENERAL TRAIN AGENT CLASS: USED COELESCE INFORMATION IN ORDER TO TRAIN
        AGENT FOLLOWING KL(P||Q) SCHEME. INCLUDES ALL SPECS GIVEN IN THE
        SETTINGS FILE, WHERE EACH OF THESE PARAMETERS CAN BE ADJUSTED. """

    def __init__(self, task):
        super(TRAIN_AGENT, self).__init__()
        # set the task being solved
        self.task = task

    def optimality(self, rewards):

        """ NORMALIZE THE EXPO OF REWARDS """
        probabilities = torch.exp(rewards) / torch.sum(torch.exp(rewards))

        """ GENERATES BERNOULLI OPTIMALITY VARIABLES """
        # sample some bernoulli rv under the distribution over probabilities
        optimality_tensor = torch.zeros((sample_size, trajectory_length, 1))
        # generate a bunch of samples
        for j in range(sample_size):
            optimality_tensor[j,:,0] = bern(probabilities).sample()
        # return
        return optimality_tensor

    def set_optimizer(self, policy):

        """ INITIALIZE CHOSEN OPTIMIZER """
        if optimize == "Adam":
            optimizer = torch.optim.Adam(policy.parameters(), lr = lr, betas = (beta_1, beta_2),
                    weight_decay = weight_decay)
            return optimizer
        elif optimize == "ASGD":
            optimizer = torch.optim.ASGD(policy.parameters(), lr = lr, lambd=lambd, alpha = alpha,
                    weight_decay = weight_decay)
            return optimizer
        elif optimize == "SGD":
            optimizer = torch.optim.SGD(policy.parameters(), lr = lr, lambd = lambd, alpha = alpha,
                    weight_decay = weight_decay)
            return optimizer
        elif optimize == "NaturalGrad":
            optimizer = NaturalGrad(policy.parameters(), lr = lr, decay = lambd, L2reg = weight_decay)
            return optimizer
        else:
            print("optim not supported sorry.")
            return None

    def set_policy(self):

        """ SET THE NUERAL NETWORK MODEL USED FOR THE CURRENT PROBLEM """
        if agent_model == 'DISCRETE':
            return RWS_DISCRETE_POLICY(state_size, actions, trajectory_length, hidden_layer_size)
        elif agent_model == 'CONTINUOUS':
            return RWS_CONTINUOUS_POLICY(state_size, actions, trajectory_length)
        elif agent_model == 'DISCRETE_TWISTED':
            return RWS_DISCRETE_TWISTED_POLICY(state_size, actions, trajectory_length)
        elif agent_model == 'CONTINUOUS_TWISTED':
            return RWS_CONTINUOUS_TWISTED_POLICY(state_size, actions, trajectory_length)
        else:
            print("error: brah.")

    def train_gym_task(self):

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        # initialize env
        env = gym.make(self.task)
        # set game
        game = GAME_SAMPLES(state_size, action_size, self.task, trajectory_length, \
                    sample_size, action_transform, reward_shaping)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((iterations))
        # time per iteration
        time_per_iteration = torch.zeros((iterations))
        # non zero importance weights
        nonzeroiw_per_iteration = torch.zeros((iterations))
        # time for project info
        begining = time.time()

        """ INITIALIZE OBJECTIVE, POLICY, AND OPTIMIZER """
        # add probs
        probabilities = torch.ones((trajectory_length))
        # add loss module
        iwloss =  SIMPLE_WAKE()
        # add model
        policy = self.set_policy()
        # optimization method
        optimizer = self.set_optimizer(policy)

        """ TRAIN AGENT """
        for iter in range(iterations):

            """ SAMPLE FROM SIMULATOR ASSUMING OPTIMALITY """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total = game.sample_game(env, policy, torch.ones(sample_size, trajectory_length))

            """ SAMPLE OPTIMALITY VARIABLES USING SELF NORMALIZED IMPORTANCE WEIGHTING """
            optim = self.optimality(reward_total.squeeze(2))

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # initialize pytorch dataset for efficient data loading
            data = torch.utils.data.TensorDataset(state_total, action_total.long(), reward_total, optim)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=workers)

            # update model on gpu following mini-batches
            for r, (state_batch, action_batch, reward_batch, optim_batch) in enumerate(data_loader):
                # compute loss optim_tensor, state_tensor, action_tensor, reward_tensor, policy
                loss = iwloss(optim_batch, state_batch, action_batch, reward_batch, policy)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

            """ RE-RUN EXPERIMENTS UNDER OPTIMALITY TO SEE HOW WE ARE DOING """
            _, _, reward_total = game.sample_game(env, policy, torch.ones(optim.size()))
            exp = torch.sum(torch.sum(inv_reward_shaping(reward_total),1)/sample_size)

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp

            """ PRINT STATEMENTS """
            if iter % floor((iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(exp))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration
