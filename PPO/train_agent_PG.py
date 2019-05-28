
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


class PG_CARTPOLE(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size, num_workers, adam_params):
        super(PG_CARTPOLE, self).__init__()
        self.task = task
        self.iterations = iterations
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # DataLoader info
        self.batch_size = batch_size
        self.num_workers = num_workers
        # optim paramaters
        self.lr = adam_params['lr']
        self.betas = adam_params['betas']
        self.weight_decay = adam_params['weight_decay']

    def train_gym_task(self):
        """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """
        # initialize env to get info
        env = gym.make(self.task)
        # get state and action dimensions from enviroment
        action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
        state_size = 4 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 2 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        # reward_shaping = lambda r: 5*r - 6
        # rev_reward_shaping = lambda r: (r+6)/5
        reward_shaping = lambda r: r
        rev_reward_shaping = lambda r: r
        # set game
        game = PG_CARTPOLE_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size, reward_shaping)
        # initial policy
        policy = PG_DISCRETE_POLICY(state_size, action_size, actions)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, betas = self.betas, weight_decay= self.weight_decay)
        # loss function
        pgloss = PG_LOSS(self.trajectory_length, self.sample_size)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))
        # set start time
        begining = time.time()

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE FROM SIMULATOR """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total, cumsum_total = game.sample_game(env, policy)

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # initialize pytorch dataset for efficient data loading
            data = torch.utils.data.TensorDataset(state_total, action_total.long(), \
                                                  reward_total, cumsum_total)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            """ MINI-BATCH UPDATES """
            # update model on gpu following mini-batches
            for r, (state_batch, action_batch, reward_batch, cumsum_batch) in enumerate(data_loader):
                # compute loss
                loss = pgloss(policy, state_batch, action_batch, reward_batch, cumsum_batch)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

            """ UPDATE DATA STORAGE """
            end = time.time()
            time_per_iteration[iter] = end - start
            expected_reward = torch.sum(torch.sum(rev_reward_shaping(reward_total),1)/self.sample_size)
            loss_per_iteration[iter] = expected_reward


            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(expected_reward))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print("Total elapsed time: " + str(end - begining) + ' at iteration: ' + str(iter))
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration

class PG_PENDULUM(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size, num_workers, adam_params):
        super(PG_PENDULUM, self).__init__()
        self.task = task
        self.iterations = iterations
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # DataLoader info
        self.batch_size = batch_size
        self.num_workers = num_workers
        # optim paramaters
        self.lr = adam_params['lr']
        self.betas = adam_params['betas']
        self.weight_decay = adam_params['weight_decay']

    def train_gym_task(self):

        """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """
        # initialize env to get info
        env = gym.make(self.task)
         # get state and action dimensions from enviroment
        action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
        state_size = 3 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        action_box = torch.tensor([-2.0, 2.0])
        reward_shaping = lambda r: r
        rev_reward_shaping = lambda r: r

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        # set up game samples
        game = PG_PENDULUM_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size)
        # loss function
        pgloss = PG_LOSS(self.trajectory_length, self.sample_size)
        # add model
        policy = PG_CONT_POLICY(state_size, action_size, self.trajectory_length)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, betas = self.betas, weight_decay= self.weight_decay)
        # loss function
        pgloss = PG_LOSS(self.trajectory_length, self.sample_size)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))
        # set start time
        begining = time.time()

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE FROM SIMULATOR """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total, cumsum_total = game.sample_game(env, policy)

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # initialize pytorch dataset for efficient data loading
            data = torch.utils.data.TensorDataset(state_total, action_total.long(), \
                                                  reward_total, cumsum_total)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            """ MINI-BATCH UPDATES """
            # update model on gpu following mini-batches
            for r, (state_batch, action_batch, reward_batch, cumsum_batch) in enumerate(data_loader):
                # compute loss
                loss = pgloss(policy, state_batch, action_batch, reward_batch, cumsum_batch)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

            """ UPDATE DATA STORAGE """
            end = time.time()
            time_per_iteration[iter] = end - start
            expected_reward = torch.sum(torch.sum(rev_reward_shaping(reward_total),1)/self.sample_size)
            loss_per_iteration[iter] = expected_reward


            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(expected_reward))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print("Total elapsed time: " + str(end - begining) + ' at iteration: ' + str(iter))
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration

class PG_MOUNTAINCAR(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size = 1, num_workers = 2, normalize = 1):
        super(MOUNTAINCAR, self).__init__()
        self.task = task
        self.iterations = iterations
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # DataLoader info
        self.batch_size = batch_size
        self.num_workers = num_workers
        # normalize
        self.normalize = normalize

    def optimality(self, probabilities):
        # sample some bernoulli rv under the distribution over probabilities
        optimality_tensor = torch.zeros((self.sample_size, self.trajectory_length, 1))
        for t in range(self.trajectory_length):
            for j in range(self.sample_size ):
                optim_dist = Bernoulli(probabilities[t])
                optimality_tensor[j, t, 0] = optim_dist.sample()
        # return
        return optimality_tensor

    def train_gym_task(self, optim_probabilities):

        """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """
        # initialize env to get info
        env = gym.make(self.task)
        # get state and action dimensions from enviroment
        action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
        state_size = 2 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 3 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        # initialize game
        game = MOUNTAIN_CAR_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size)
        # initial policy
        policy = DISCRETE_POLICY(state_size, action_size, actions)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
        # loss function
        pgloss = PG_LOSS(self.trajectory_length, self.sample_size)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))
        # set start time
        begining = time.time()

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE FROM SIMULATOR """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total, cumsum_total = game.sample_game(env, policy)

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # initialize pytorch dataset for efficient data loading
            data = torch.utils.data.TensorDataset(state_total, action_total.long(), \
                                                  reward_total, cumsum_total)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

            """ MINI-BATCH UPDATES """
            # update model on gpu following mini-batches
            for r, (state_batch, action_batch, reward_batch, cumsum_batch) in enumerate(data_loader):
                # compute loss
                loss = pgloss(policy, state_batch, action_batch, reward_batch, cumsum_batch)
                # backprop through computation graph
                loss.backward()

            """ STEP OPTIMIZER AND ZERO OUT GRADIENT """
            # step optimizer
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            """ UPDATE DATA STORAGE """
            end = time.time()
            time_per_iteration[iter] = end - start
            expected_reward = torch.sum(cumsum_total[:,0]) / self.sample_size
            loss_per_iteration[iter] = expected_reward

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(expected_reward))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print("Total elapsed time: " + str(end - begining) + ' at iteration: ' + str(iter))
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration
