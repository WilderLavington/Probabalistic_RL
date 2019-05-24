
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
from NaturalGrad import *

class MOUNTAINCAR(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size, num_workers, \
                    normalize, params, include_buffer, buffer_size, optimizer, buffer_update_type, \
                    sample_reg, apply_filtering, trust_region_reg, approx_lagrange):
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
        # set optimizer
        self.optimizer = optimizer
        self.optim_params = params
        # replay buffer info
        self.include_buffer = include_buffer
        if include_buffer == 1:
            if buffer_size:
                self.buffer_size = torch.tensor(buffer_size).float()
            else:
                self.buffer_size = torch.tensor(sample_size).float()
            # other info
            self.buffer_states = None
            self.buffer_action = None
            self.buffer_reward = None
            self.buffer_optim = None
            self.buffer_set = False
        # set buffer update type
        self.buffer_update_type = buffer_update_type
        self.sample_reg = sample_reg
        self.apply_filtering = apply_filtering
        self.trust_region_reg = trust_region_reg
        self.approx_lagrange = approx_lagrange

    def optimality(self, probabilities):
        # sample some bernoulli rv under the distribution over probabilities
        optimality_tensor = torch.zeros((self.sample_size, self.trajectory_length, 1))
        # generate a bunch of samples
        for j in range(self.sample_size ):
            optimality_tensor[j,:,0] = bern(probabilities).sample()
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
        reward_shaping = lambda r: r
        rev_reward_shaping = lambda r: r
        game = MOUNTAIN_CAR_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))
        # non zero importance weights
        nonzeroiw_per_iteration = torch.zeros((self.iterations))
        # time for project info
        begining = time.time()

        """ INITIALIZE WHAT WE CAN ON THE GPU """
        # add probs
        probabilities = optim_probabilities
        # add loss module
        iwloss = IW_WAKE(self.trajectory_length, self.sample_size, probabilities, self.normalize, rev_reward_shaping, self.buffer_update_type, \
                self.sample_reg, self.apply_filtering, self.trust_region_reg, self.approx_lagrange)
        # add model
        policy = RWS_DISCRETE_POLICY(state_size, actions, self.trajectory_length)

        """ INITIALIZE CHOSEN OPTIMIZER """
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(policy.parameters(), lr=self.optim_params["lr"], betas = self.optim_params["betas"],
                     weight_decay = self.optim_params["weight_decay"])
        elif self.optimizer == "ASGD":
            optimizer = torch.optim.ASGD(policy.parameters(), lr=self.optim_params["lr"], lambd = self.optim_params["lambd"], alpha = self.optim_params["alpha"],
                     weight_decay = self.optim_params["weight_decay"])
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(policy.parameters(), lr=self.optim_params["lr"], lambd = self.optim_params["lambd"], alpha = self.optim_params["alpha"],
                     weight_decay = self.optim_params["weight_decay"])
        elif self.optimizer == "NaturalGrad":
            optimizer = NaturalGrad(policy.parameters(), lr=self.optim_params["lr"], lambd = self.optim_params["lambd"],
                     weight_decay = self.optim_params["weight_decay"])
        else:
            print("optim not supported sorry.")
            return None

        """ SET BUFFER UPDATE SO WE START STORING OLD SAMPLES """
        if self.include_buffer:
            # set the buffer to be used
            iwloss.set_buffer_size(self.buffer_size)
            iwloss.start_buffer_updates(True)

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE OPTIMALITY VARIABLES """
            optim = self.optimality(probabilities)

            """ SAMPLE FROM SIMULATOR """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total = game.sample_game(env, policy, optim)

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # initialize pytorch dataset for efficient data loading
            data = torch.utils.data.TensorDataset(state_total, action_total.long(), reward_total, optim)
            # set data loader
            data_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            # set average to so current expectation
            exp_iw = 0
            counter = 0
            # update model on gpu following mini-batches
            for r, (state_batch, action_batch, reward_batch, optim_batch) in enumerate(data_loader):
                # compute loss
                loss, nonzero_iw, samples = iwloss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()
                # update the importance weights
                exp_iw += nonzero_iw
                counter += 1

            """ RE-RUN EXPERIMENTS UNDER OPTIMALITY TO SEE HOW WE ARE DOING """
            _, _, reward_total = game.sample_game(env, policy, torch.ones(optim.size()))
            exp = torch.sum(torch.sum(rev_reward_shaping(reward_total),1)/self.sample_size)

            """ START USING REPLAY BUFFER NOW THAT WE HAVE SAMPLES """
            if self.include_buffer:
                # set the buffer to be used
                iwloss.use_buffer(True)

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp
            nonzeroiw_per_iteration[iter] = exp_iw / counter

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(exp))
                print("Average number of non-zero importance weights: " + str(exp_iw / counter) + " Out of " + str(samples))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration
