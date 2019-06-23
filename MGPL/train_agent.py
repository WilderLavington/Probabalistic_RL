
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
from objective_function import *
from train_agent import *
from agent_models import *
from transition_dyna_models import *
from gen_reward_models import *
from rl_variable_imports import *

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

    def set_optimizer(self, model):

        """ INITIALIZE CHOSEN OPTIMIZER """
        if optimize == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas = (beta_1, beta_2),
                    weight_decay = weight_decay)
            return optimizer
        elif optimize == "ASGD":
            optimizer = torch.optim.ASGD(model.parameters(), lr = lr, lambd=lambd, alpha = alpha,
                    weight_decay = weight_decay)
            return optimizer
        elif optimize == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = lr, lambd = lambd, alpha = alpha,
                    weight_decay = weight_decay)
            return optimizer
        elif optimize == "NaturalGrad":
            optimizer = NaturalGrad(model.parameters(), lr = lr, decay = lambd, L2reg = weight_decay)
            return optimizer
        else:
            print("optim not supported sorry.")
            return None

    def set_policy(self):

        """ SET THE NUERAL NETWORK AGENT MODEL USED FOR THE CURRENT PROBLEM """
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

    def set_dyna_model(self):

        """ SET THE NUERAL NETWORK DYNAMICS MODEL USED FOR THE CURRENT PROBLEM """
        if agent_model == 'DISCRETE':
            return TRANSITION_DYNAMICS_MODEL_DESC(state_size, actions, trajectory_length, hidden_layer_size)
        elif agent_model == 'CONTINUOUS':
            return TRANSITION_DYNAMICS_MODEL_CONT(state_size, actions, trajectory_length)
        else:
            print("error: brah.")

    def set_reward_model(self):

        """ SET THE NUERAL NETWORK DYNAMICS MODEL USED FOR THE CURRENT PROBLEM """
        if agent_model == 'DISCRETE':
            return TRANSITION_REWARD_MODEL_DESC(state_size, actions, trajectory_length, hidden_layer_size)
        elif agent_model == 'CONTINUOUS':
            return TRANSITION_REWARD_MODEL_CONT(state_size, actions, trajectory_length)
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

        """ INITIALIZE OBJECTIVE FOR DYNAMICS MODEL, REWARD MODEL, AND INFERENCE NETWORK """
        # add probs
        probabilities = optim_probabilities
        # add loss module (generative models)
        GenRewardloss =  THETA_WAKE_REWARD()
        GenDynaloss =  THETA_WAKE_DYNA()
        # inference models - need two bc of replay buffer
        InfSleeploss = PHI_SLEEP(trajectory_length, batch_size, probabilities, normalize, inv_reward_shaping, buffer_update_type, \
                            sample_reg, trust_region_reg, approx_lagrange, use_running_avg, running_avg_norm, running_avg_count)
        InfWakeloss = PHI_WAKE(trajectory_length, batch_size, probabilities, normalize, inv_reward_shaping, buffer_update_type, \
                            sample_reg, trust_region_reg, approx_lagrange, use_running_avg, running_avg_norm, running_avg_count)

        """ INITIALIZE MODEL FOR DYNAMICS MODEL, REWARD MODEL, AND INFERENCE NETWORK """
        # add models
        policy = self.set_policy()
        dyna_model = self.set_dyna_model()
        reward_model = self.set_reward_model()

        """ INITIALIZE OPTIMIZER FOR DYNAMICS MODEL, REWARD MODEL, AND INFERENCE NETWORK """
        # optimization method for parameters
        optimizer_infNet = self.set_optimizer(policy)
        optimizer_genRewardNet = self.set_optimizer(reward_model)
        optimizer_genDynaNet = self.set_optimizer(dyna_model)

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
            # update model following mini-batches
            for r, (state_batch, action_batch, reward_batch, optim_batch) in enumerate(data_loader):

                """ WAKE PHASE - THETA UPDATE - REWARD UPDATE """
                # now apply the update as usual
                loss = GenRewardloss(reward_model, state_batch, action_batch, reward_batch, optim_batch)
                # zero the parameter gradients
                optimizer_genRewardNet.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

                """ WAKE PHASE - THETA UPDATE - DYNAMICS UPDATE """
                # now apply the update as usual
                loss = GenDynaloss(dyna_model, state_batch, action_batch, reward_batch, optim_batch)
                # zero the parameter gradients
                optimizer_genDynaNet.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

                """ SLEEP PHASE - PHI UPDATE """
                # generate fake data using the generative model
                gen_state_batch, gen_action_batch, gen_reward_batch, gen_optim_batch = model_based_simulator(policy, dyna_model, reward_model, optim)
                # now apply the update as usual
                loss, nonzero_iw, samples = InfSleeploss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # zero the parameter gradients
                optimizer_infNet.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()

                """ WAKE PHASE - PHI UPDATE """
                # compute loss
                loss, nonzero_iw, samples = InfWakeloss(policy, state_batch, action_batch, reward_batch, optim_batch)
                # zero the parameter gradients
                optimizer_infNet.zero_grad()
                # backprop through computation graph
                loss.backward()
                # step optimizer
                optimizer.step()
                # update the importance weights
                exp_iw += nonzero_iw
                counter += 1

            """ RE-RUN EXPERIMENTS UNDER OPTIMALITY TO SEE HOW WE ARE REALLY DOING """
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

    
