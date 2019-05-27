
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

""" SET GAME TYPE VIA IMPORT SETTINGS """
# from settings_CARTPOLE import *
from settings_ACROBOT import *
# from settings_MOUNTAINCAR_DISC import *
# from settings_PENDULUM import *

class TRAIN_AGENT(torch.nn.Module):

    """ GENERAL TRAIN AGENT CLASS: USED COELESCE INFORMATION IN ORDER TO TRAIN
        AGENT FOLLOWING KL(P||Q) SCHEME. INCLUDES ALL SPECS GIVEN IN THE
        SETTINGS FILE, WHERE EACH OF THESE PARAMETERS CAN BE ADJUSTED. """

    def __init__(self, task):
        super(TRAIN_AGENT, self).__init__()
        # set the task being solved
        self.task = task

    def optimality(self, probabilities):

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
            return RWS_DISCRETE_POLICY(state_size, actions, trajectory_length)
        elif agent_model == 'CONTINUOUS':
            return RWS_CONTINUOUS_POLICY(state_size, actions, trajectory_length)
        elif agent_model == 'DISCRETE_TWISTED':
            return RWS_DISCRETE_TWISTED_POLICY(state_size, actions, trajectory_length)
        elif agent_model == 'CONTINUOUS_TWISTED':
            return RWS_CONTINUOUS_TWISTED_POLICY(state_size, actions, trajectory_length)
        else:
            print("error: brah.")

    def train_gym_task(self, optim_probabilities):

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
        probabilities = optim_probabilities
        # add loss module
        iwloss =  IW_WAKE(trajectory_length, batch_size, probabilities, normalize, inv_reward_shaping, buffer_update_type, \
                            sample_reg, trust_region_reg, approx_lagrange)
        # add model
        policy = self.set_policy()
        # optimization method
        optimizer = self.set_optimizer(policy)

        """ SET BUFFER UPDATE SO WE START STORING OLD SAMPLES """
        if include_buffer == 1:
            # set the buffer to be used
            iwloss.set_buffer_size(buffer_size)
            iwloss.start_buffer_updates(True)

        """ TRAIN AGENT """
        for iter in range(iterations):

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
            data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=workers)
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
            exp = torch.sum(torch.sum(inv_reward_shaping(reward_total),1)/sample_size)

            """ START USING REPLAY BUFFER NOW THAT WE HAVE SAMPLES """
            if include_buffer:
                # set the buffer to be used
                iwloss.use_buffer(True)

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp
            nonzeroiw_per_iteration[iter] = exp_iw / counter

            """ PRINT STATEMENTS """
            if iter % floor((iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(exp))
                print("Average number of non-zero importance weights: " + str(exp_iw / counter) + " Out of " + str(samples))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration

    def train_gym_task_adaptive(self, optim_probabilities_init, adaptive_step):

        """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """
        # initialize env to get info
        env = gym.make(self.task)
        # get state and action dimensions from enviroment
        action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
        state_size = 4 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 2 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        reward_shaping = lambda r: 5*r - 6
        rev_reward_shaping = lambda r: (r+6)/5
        set_to_one = 20
        game = RWS_CARTPOLE_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size, reward_shaping)

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # time per iteration
        time_per_iteration = torch.zeros((self.iterations))
        # non zero importance weights
        nonzeroiw_per_iteration = torch.zeros((self.iterations))
        # time for project info
        begining = time.time()

        """ INITIALIZE WHAT WE CAN """
        # add probs
        probabilities = optim_probabilities_init
        # add loss module
        iwloss = IW_WAKE(self.trajectory_length, self.sample_size, probabilities, self.normalize, rev_reward_shaping, self.buffer_update_type, \
                self.sample_reg, self.apply_filtering, self.trust_region_reg, self.approx_lagrange)
        # add model
        policy = RWS_DISCRETE_POLICY(state_size, actions, self.trajectory_length)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, rho=self.rho, weight_decay=self.weight_decay)

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
                loss, _, nonzero_iw = iwloss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()
                # update the importance weights
                exp_iw += nonzero_iw
                counter += 1

            # compute optimal expected values
            _, _, reward_total = game.sample_game(env, policy, torch.ones(optim.size()))
            exp = torch.sum(torch.sum(rev_reward_shaping(reward_total),1)/self.sample_size)

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp
            nonzeroiw_per_iteration[iter] = exp_iw / counter

            """ UPDATE OPTIMALITY VARIABLES """
            current_exp_wins = exp.long()
            set_to_one = max(set_to_one, current_exp_wins+adaptive_step)
            probabilities[0:min(set_to_one, len(probabilities))] = 1.

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(exp))
                print("Average number of non-zero importance weights: " + str(exp_iw / counter) + " Out of " + str(self.batch_size))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration







class SEQ_LEARNING(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size = 1, num_workers = 2, normalize = 1):
        super(SEQ_LEARNING, self).__init__()
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
        state_size = 1 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 3 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        reward_shaping = lambda r: 5*r - 6
        rev_reward_shaping = lambda r: (r+6)/5
        game = SEQ_GEN(state_size, action_size, self.task, self.trajectory_length, self.sample_size)

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
        iwloss = IW_WAKE(self.trajectory_length, self.sample_size, probabilities, self.normalize)
        # add model
        policy = RWS_DISCRETE_POLICY(state_size, actions, self.trajectory_length)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

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
                loss, _, nonzero_iw = iwloss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # backprop through computation graph
                loss.backward()
                # update the importance weights
                exp_iw += nonzero_iw
                counter += 1

            # compute optimal expected values
            _, _, reward_total = game.sample_game(env, policy, torch.ones(optim.size()))
            exp = torch.sum(torch.sum(rev_reward_shaping(reward_total),1)/self.sample_size)

            """ STEP OPTIMIZER AND ZERO OUT GRADIENT """
            # step optimizer
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp
            nonzeroiw_per_iteration[iter] = exp_iw / counter

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(exp))
                print("Average number of non-zero importance weights: " + str(exp_iw / counter) + " Out of " + str(self.batch_size))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration

class TRAIN_HAIL_MARY(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size = 1, num_workers = 2, normalize = 1, increase_optim = True):
        super(TRAIN_HAIL_MARY, self).__init__()
        self.task = task
        self.iterations = iterations
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # DataLoader info
        self.batch_size = batch_size
        self.num_workers = num_workers
        # normalize
        self.normalize = normalize
        # adaptive
        self.increase_optim = increase_optim

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
        state_size = 4 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 2 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        game = CARTPOLE_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size)

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
        iwloss = IW_WAKE(self.trajectory_length, self.sample_size, probabilities, self.normalize)
        # add model
        policy = HAIL_MARY(state_size, actions, self.trajectory_length)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

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
                loss, _, nonzero_iw = iwloss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # backprop through computation graph
                loss.backward()
                # update the importance weights
                exp_iw += nonzero_iw
                counter += 1

            # compute optimal expected values
            _, _, reward_total = game.sample_game(env, policy, torch.ones(optim.size()))
            exp = torch.sum(torch.sum((reward_total+2),1)/self.sample_size)

            """ STEP OPTIMIZER AND ZERO OUT GRADIENT """
            # step optimizer
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp
            nonzeroiw_per_iteration[iter] = exp_iw / counter

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:

                print("Expected Sum of rewards: " + str(exp))
                print("Average number of non-zero importance weights: " + str(exp_iw / counter) + " Out of " + str(self.batch_size))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration

class SEQ_CARTPOLE(torch.nn.Module):

    def __init__(self, task, iterations, sample_size, trajectory_length, batch_size = 1, num_workers = 2, normalize = 1):
        super(SEQ_CARTPOLE, self).__init__()
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

    def optimality_batch(self, probabilities):
        # sample some bernoulli rv under the distribution over probabilities
        optimality_tensor = torch.zeros((self.sample_size, self.trajectory_length, self.trajectory_length))
        for j in range(self.sample_size):
            optim_temp = torch.zeros(self.trajectory_length)
            for t in range(self.trajectory_length):
                optim_dist = Bernoulli(probabilities[t])
                optim_temp[t] = optim_dist.sample()
            # set the whole thing in as an input
            optimality_tensor[j,:,:] = optim_temp
        # return
        return optimality_tensor

    def train_gym_task(self, optim_probabilities):

        """ TRAIN AGENT TO COMPLETE GYM TASK VIA BASELINE ADJUSTED POLICY GRADIENTS """
        # initialize env to get info
        env = gym.make(self.task)
        # get state and action dimensions from enviroment
        action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
        state_size = 4 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
        actions = 2 # env.action_space.n

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        game = CARTPOLE_GAME_SAMPLES(state_size, action_size, self.task, self.trajectory_length, self.sample_size)

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
        iwloss = SEQ_IW_WAKE(self.trajectory_length, self.sample_size, probabilities, self.normalize)
        # add model
        policy = SEQ_DISCRETE_POLICY(state_size, actions, self.trajectory_length)
        # initialize optimizer
        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE OPTIMALITY VARIABLES """
            optim = self.optimality(probabilities)

            """ SAMPLE FROM SIMULATOR """
            # time for project info
            start = time.time()
            # get new batch of trajectories from simulator
            state_total, action_total, reward_total = game.sample_game_seq(env, policy, optim)

            """ RUN MULTIPLE UPDATES ON GPU USING MINI-BATCHES """
            # populate optim batch
            optim = self.optimality_batch(probabilities)
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
                loss, _, nonzero_iw = iwloss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # backprop through computation graph
                loss.backward()
                # update the importance weights
                exp_iw += nonzero_iw
                counter += 1

            # compute optimal expected values
            _, _, reward_total = game.sample_game_seq(env, policy, torch.ones(reward_total.size()))
            exp = torch.sum(torch.sum((reward_total+4)/3,1)/self.sample_size)

            """ STEP OPTIMIZER AND ZERO OUT GRADIENT """
            # step optimizer
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            """ UPDATE DATA STORAGE """
            # update time
            end = time.time()
            time_per_iteration[iter] = end - start
            loss_per_iteration[iter] = exp
            nonzeroiw_per_iteration[iter] = exp_iw / counter

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("Expected Sum of rewards: " + str(exp))
                print("Average number of non-zero importance weights: " + str(exp_iw / counter) + " Out of " + str(self.batch_size))
                print("Loss: " + str(loss))
                print('Time per Iteration: ' + str(end - start) + ' at iteration: ' + str(iter))
                print('Time elapsed: ' + str((end - begining)/60) + ' minutes.')
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, time_per_iteration, nonzeroiw_per_iteration
