
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

class IW_WAKE(torch.nn.Module):

    """ POLICY GRADIENTS LOSS FUNCTION """
    def __init__(self, trajectory_length, simulations, probabilities, normalize, rev_reward_shaping, buffer_update_type, \
                sample_reg, apply_filtering, trust_region_reg, approx_lagrange):
        """ INITIALIZATIONS """
        super(IW_WAKE, self).__init__()
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.probabilities = probabilities
        self.action_size = 2
        self.epsilon = 0.0000001
        self.normalize = normalize
        # buffer replay info
        self.buffer_size = 0
        self.current_buffer_size = 0
        self.buffer_states = None
        self.buffer_action = None
        self.buffer_reward = None
        self.buffer_optim = None
        self.buffer_weights = None
        # should it be included
        self.include_buffer = False
        self.begin_buffer_updates = False
        self.buffer_set = False
        # old_policy to generate importance weights
        self.old_policy = None
        # inverse reward shaping to order solutions
        self.rev_reward_shaping = rev_reward_shaping
        # choose buffer update type
        self.buffer_update_type = buffer_update_type
        self.sample_reg = trajectory_length / 10
        # should we filter samples prior to update
        self.apply_filtering = apply_filtering
        # use trust region method
        self.trust_region_reg = trust_region_reg
        # value for approximate lagrange multiplier
        self.approx_lagrange = approx_lagrange
        # stored KL reg
        self.KL = None

    def use_buffer(self, use):
        self.include_buffer = use
        self.buffer_set = True
        return None

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size.int()
        return None

    def start_buffer_updates(self, use):
        self.begin_buffer_updates = use
        return None

    """ SAMPLE BASED APPROACH BASED UPON NORMALIZED REWARDS BETWEEN ALL TRAJECTORIES """
    def update_buffer_RS(self, new_states, new_actions, new_rewards, new_optim, current_policy, fresh_buffer):
        # set the current buffer size
        buffer_size = self.buffer_size
        # if the buffer is currently empty replace it sorted states
        if not self.buffer_set:
            # compute cumulative rewards through time for each sample
            new_cum_rewards = torch.sum(new_rewards, dim=1)
            # replace cumulative reward by probability
            if len(new_cum_rewards) == 1:
                prob_cum_rewards = torch.tensor([1.0])
            else:
                prob_cum_rewards = (new_cum_rewards - torch.min(new_cum_rewards) + self.sample_reg) / torch.sum(new_cum_rewards - torch.min(new_cum_rewards) + self.sample_reg)
            # set Categorical distributions
            buffer_distribution = Categorical(prob_cum_rewards[0])
            samples = [buffer_distribution.sample() for i in range(self.buffer_size)]
            # set old policy
            self.old_policy = current_policy
            # store info
            self.buffer_states = new_states[torch.stack(samples),:,:]
            self.buffer_action = new_actions[torch.stack(samples),:,:]
            self.buffer_reward = new_rewards[torch.stack(samples),:,:]
            self.buffer_optim = new_optim[torch.stack(samples),:]
            self.buffer_set = True
            self.current_buffer_size = self.buffer_size
        else:
            # set update the old policy
            self.old_policy = current_policy
            # do we want to throw away the old buffer and just sample from the new data
            if fresh_buffer:
                # combine it with the current buffer
                self.buffer_states = new_states
                self.buffer_action = new_actions
                self.buffer_reward = new_rewards
                self.buffer_optim = new_optim
            else:
                # combine it with the current buffer
                self.buffer_states = torch.cat([new_states, self.buffer_states])
                self.buffer_action = torch.cat([new_actions, self.buffer_action])
                self.buffer_reward = torch.cat([new_rewards, self.buffer_reward])
                self.buffer_optim = torch.cat([new_optim, self.buffer_optim])
            # compute the new cumulative rewards
            new_cum_rewards = torch.sum(self.buffer_reward, dim=1)
            # replace cumulative reward by probability
            if torch.sum(new_cum_rewards - torch.min(new_cum_rewards)) != 0:
                prob_cum_rewards = (new_cum_rewards - torch.min(new_cum_rewards) + self.sample_reg) / torch.sum(new_cum_rewards - torch.min(new_cum_rewards) + self.sample_reg)
                prob_cum_rewards = prob_cum_rewards.reshape(-1)
            else:
                prob_cum_rewards = torch.ones(new_cum_rewards.size()) / len(torch.ones(new_cum_rewards.size()))
                prob_cum_rewards = prob_cum_rewards.reshape(-1)
            # set Categorical distributions
            buffer_distribution = Categorical(prob_cum_rewards)

            samples = [buffer_distribution.sample() for i in range(self.buffer_size)]
            # store info
            self.buffer_states = self.buffer_states[torch.stack(samples),:,:]
            self.buffer_action = self.buffer_action[torch.stack(samples),:,:]
            self.buffer_reward = self.buffer_reward[torch.stack(samples),:,:]
            self.buffer_optim = self.buffer_optim[torch.stack(samples),:]
            self.buffer_set = True
            self.current_buffer_size = self.buffer_size
            # print info
            print("Current Average Buffer Value: " + str(torch.mean(torch.sum(self.rev_reward_shaping(self.buffer_reward), dim=1))))
            # print(len(self.buffer_weights))

        """ COMPUTE AND STORE KL BETWEEN SAMPLING DISTRIBUTION AND POLICY """
        if self.trust_region_reg == 1:
            p = prob_cum_rewards[torch.stack(samples)]
            flat_states = torch.flatten(self.buffer_states, start_dim=0,end_dim=1)
            flat_actions = torch.flatten(self.buffer_action, start_dim=0,end_dim=1)
            flat_opt = torch.flatten(self.buffer_optim, start_dim=0,end_dim=1)
            log_p = torch.log(p + self.epsilon)
            log_q = torch.sum(current_policy(flat_states, flat_actions, flat_opt).reshape(self.buffer_reward.size()).squeeze(2), dim=1)
            self.KL = torch.sum(p*(log_p - log_q))

        # return buffer info
        return self.buffer_states, self.buffer_action, self.buffer_reward, self.buffer_optim, buffer_distribution

    """ SAMPLE BASED APPROACH FOLLOWING THE IMPORTANCE WEIGHTS OF THE ORIGNAL DISTRIBUTION """
    def update_buffer_IWS(self, new_states, new_actions, new_rewards, new_optim, fresh_buffer, current_policy, iw):
        # set the current buffer size
        buffer_size = self.buffer_size
        # if the buffer is currently empty replace it sorted states
        if not self.buffer_set:
            # replace cumulative reward by probability
            if len(iw) == 1:
                prob_cum_rewards = torch.tensor([1.0])
            else:
                prob_cum_rewards = iw / torch.sum(iw)
            # set Categorical distributions
            buffer_distribution = Categorical(prob_cum_rewards)
            samples = [buffer_distribution.sample() for i in range(self.buffer_size)]
            # set old policy
            self.old_policy = current_policy
            # store info
            self.buffer_states = new_states[samples,:,:]
            self.buffer_action = new_actions[samples,:,:]
            self.buffer_reward = new_rewards[samples,:,:]
            self.buffer_optim = new_optim[samples,:]
            self.buffer_iw = iw[samples]
            self.buffer_set = True
            self.current_buffer_size = self.buffer_size
        else:
            # set update the old policy
            self.old_policy = current_policy
            # do we want to throw away the old buffer and just sample from the new data
            if fresh_buffer:
                # combine it with the current buffer
                self.buffer_states = new_states
                self.buffer_action = new_actions
                self.buffer_reward = new_rewards
                self.buffer_optim = new_optim
            else:
                # combine it with the current buffer
                self.buffer_states = torch.cat([new_states, self.buffer_states])
                self.buffer_action = torch.cat([new_actions, self.buffer_action])
                self.buffer_reward = torch.cat([new_rewards, self.buffer_reward])
                self.buffer_optim = torch.cat([new_optim, self.buffer_optim])
                self.buffer_iw = torch.cat([iw, self.buffer_iw])

            # replace cumulative reward by probability
            prob_cum_rewards = iw / torch.sum(iw)
            # set Categorical distributions
            buffer_distribution = Categorical(prob_cum_rewards)
            samples = [buffer_distribution.sample() for i in range(self.buffer_size)]
            # store info
            self.buffer_states = self.buffer_states[samples,:,:]
            self.buffer_action = self.buffer_action[samples,:,:]
            self.buffer_reward = self.buffer_reward[samples,:,:]
            self.buffer_optim = self.buffer_optim[samples,:]
            self.buffer_iw = self.buffer_iw[samples]
            self.buffer_set = True
            self.current_buffer_size = self.buffer_size
            # print info
            print("Current Average Buffer Value: " + str(torch.mean(torch.sum(self.rev_reward_shaping(self.buffer_reward), dim=1))))
            # print(len(self.buffer_weights))

        """ COMPUTE AND STORE KL BETWEEN SAMPLING DISTRIBUTION AND POLICY """
        if self.trust_region_reg == 1:
            p = prob_cum_rewards[torch.stack(samples)]
            flat_states = torch.flatten(self.buffer_states, start_dim=0,end_dim=1)
            flat_actions = torch.flatten(self.buffer_action, start_dim=0,end_dim=1)
            flat_opt = torch.flatten(self.buffer_optim, start_dim=0,end_dim=1)
            log_p = torch.log(p + self.epsilon)
            log_q = torch.sum(current_policy(flat_states, flat_actions, flat_opt).reshape(self.buffer_reward.size()).squeeze(2), dim=1)
            self.KL = torch.sum(p*(log_p - log_q))

        # return buffer info
        return self.buffer_states, self.buffer_action, self.buffer_reward, self.buffer_optim, buffer_distribution

    """ GREEDY BUFFER UPDATE TAKING THE N HIGHEST SCORING TRAJECTORIES """
    def update_buffer_greedy(self, new_states, new_actions, new_rewards, new_optim, new_weights, current_policy):
        # compute cumulative rewards through time for each sample
        cum_rewards = torch.sum(new_rewards, dim=1)
        max_samples = len(cum_rewards)
        buffer_size = self.buffer_size
        # sort by cumulative reward
        _, idx = cum_rewards.sort()
        # if the buffer is currently empty replace it sorted states
        if not self.buffer_set:
            # set old policy
            self.old_policy = current_policy
            # store info
            self.buffer_states = torch.flatten(new_states[idx[:min(buffer_size, max_samples)],:,:], start_dim=0,end_dim=1)
            self.buffer_action = torch.flatten(new_actions[idx[:min(buffer_size, max_samples)],:,:], start_dim=0,end_dim=1)
            self.buffer_reward = torch.flatten(new_rewards[idx[:min(buffer_size, max_samples)],:,:], start_dim=0,end_dim=1)
            self.buffer_optim = torch.flatten(new_optim[idx[:min(buffer_size, max_samples)],:], start_dim=0,end_dim=1)
            self.buffer_weights = new_weights[idx[:min(buffer_size, max_samples)]].squeeze(1)
            self.buffer_set = True
            self.current_buffer_size = torch.tensor(min(buffer_size.numpy(), max_samples))
        else:
            # set update the old policy
            self.old_policy = current_policy
            # compute new info
            buffer_states = torch.flatten(new_states[idx[:min(buffer_size, max_samples)],:,:], start_dim=0,end_dim=1)
            buffer_action = torch.flatten(new_actions[idx[:min(buffer_size, max_samples)],:,:], start_dim=0,end_dim=1)
            buffer_reward = torch.flatten(new_rewards[idx[:min(buffer_size, max_samples)],:,:], start_dim=0,end_dim=1)
            buffer_optim = torch.flatten(new_optim[idx[:min(buffer_size, max_samples)],:], start_dim=0,end_dim=1)
            buffer_weights = new_weights[idx[:min(buffer_size, max_samples)]]
            # combine it with the current buffer
            self.buffer_states = torch.cat([buffer_states, self.buffer_states])
            self.buffer_action = torch.cat([buffer_action, self.buffer_action])
            self.buffer_reward = torch.cat([buffer_reward, self.buffer_reward])
            self.buffer_optim = torch.cat([buffer_optim, self.buffer_optim])
            self.buffer_weights = torch.cat([buffer_weights.squeeze(1), self.buffer_weights])
            # sort based on the cumulative rewards
            cum_rewards = torch.sum(self.rev_reward_shaping(self.buffer_reward), dim=1)
            max_samples = len(cum_rewards)
            # sort by cumulative reward
            idx = torch.argsort(cum_rewards,dim=0,descending=True)
            # remove low performing trajectories
            self.buffer_states = self.buffer_states[idx[0:min(buffer_size, max_samples)].reshape(-1),:,:]
            self.buffer_action = self.buffer_action[idx[0:min(buffer_size, max_samples)].reshape(-1),:,:]
            self.buffer_reward = self.buffer_reward[idx[0:min(buffer_size, max_samples)].reshape(-1),:,:]
            self.buffer_optim = self.buffer_optim[idx[0:min(buffer_size, max_samples)].reshape(-1),:]
            self.buffer_weights = self.buffer_weights[idx[0:min(buffer_size, max_samples)]].squeeze(1)
            # set the current size of the buffer
            self.current_buffer_size = torch.tensor(min(buffer_size.numpy(), max_samples))

            print("Current Average Buffer Value: " + str(torch.mean(torch.sum(self.rev_reward_shaping(self.buffer_reward), dim=1))))
            # print(len(self.buffer_weights))

        # return buffer info
        return self.buffer_states, self.buffer_action, self.buffer_reward, self.buffer_optim

    """ LOSS FUNCTION FOR REINFORCEMENT LEARNING AS IMPORTANCE WEIGHTED APPROXIMATE INFERENCE FOLLOWING KL(P||Q) """
    def forward(self, policy, state_tensor, action_tensor, reward_tensor, optimality_tensor):

        # compute total number of samples used in the update
        total_samples = torch.tensor(1.0*reward_tensor.size()[0])

        """ COMPUTE ADDITION OF BUFFER INFO """
        if self.include_buffer:

            # compute the number of buffer values + weights
            total_samples += self.buffer_size

            """ USE OPTIMALITY VARIABLES AS INDICATORS LOG P(TAU,O) """
            # generate sub-optimal reward matrix
            buffer_subopt_reward = torch.log(1 - torch.exp(self.buffer_reward) + self.epsilon)
            # now compute the sub-optimal matrix
            buffer_sub_opt_mat = torch.sum((1-self.buffer_optim) * buffer_subopt_reward, dim=1)
            buffer_opt_mat = torch.sum((self.buffer_optim) * self.buffer_reward, dim=1)
            # compute the total matrix for IS
            log_buffer_iw_mat = buffer_sub_opt_mat + buffer_opt_mat

            """ USE OPTIMALITY VARIABLES AS INDICATORS + COMPUTE LOG P(O) """
            opt_prob = torch.mv((self.buffer_optim.squeeze(2)), torch.log(self.probabilities + self.epsilon) )
            subopt_prob = torch.mv((1.-self.buffer_optim.squeeze(2)), torch.log(1 - self.probabilities + self.epsilon))
            logOpt = opt_prob + subopt_prob

            """ OLD POLICY SCORE FUNCTION LOG Q_OLD(TAU|O) """
            if self.old_policy == None:
                self.old_policy = policy
            # convert format to something we can feed to model
            flat_states = torch.flatten(self.buffer_states, start_dim=0,end_dim=1)
            flat_actions = torch.flatten(self.buffer_action, start_dim=0,end_dim=1)
            flat_opt = torch.flatten(self.buffer_optim, start_dim=0,end_dim=1)
            # compute the models score function
            flat_buffer_opt_scoreFxn = self.old_policy(flat_states, flat_actions, flat_opt)
            # reshapte this tensor to be time by samples
            buffer_opt_scoreFxn = flat_buffer_opt_scoreFxn.reshape(self.buffer_reward.size()).squeeze(2)

            """ ADD FRANKS THING """
            # sum accross time
            buffer_sum_opt_scoreFxn = torch.sum(buffer_opt_scoreFxn, dim=1)
            buffer_sum_opt_scoreFxn = torch.log(torch.exp(buffer_sum_opt_scoreFxn) + 1 / (self.current_buffer_size.float()))

            """ COMPUTE BUFFER IMPORTANCE WEIGHTS """
            # sum through time to get the iw
            buffer_weights = log_buffer_iw_mat.squeeze(1) - buffer_sum_opt_scoreFxn - logOpt
            # scale by the total number of samples
            buffer_weights -= torch.log(total_samples)

            """ CURRENT POLICY SCORE FUNCTION LOG Q_PHI(TAU|O) """
            # compute the models score function
            flat_buffer_opt_scoreFxn = policy(flat_states, flat_actions, flat_opt)
            # reshapte this tensor to be time by samples
            buffer_opt_scoreFxn = flat_buffer_opt_scoreFxn.reshape(self.buffer_reward.size()).squeeze(2)
            # sum accross time
            buffer_sum_opt_scoreFxn = torch.sum(buffer_opt_scoreFxn, dim=1)

        """ OLD POLICY SCORE FUNCTION LOG Q_OLD(TAU|O) """
        # convert format to something we can feed to model
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)
        flat_opt = torch.flatten(optimality_tensor, start_dim=0,end_dim=1)
        if self.old_policy == None:
            self.old_policy = policy
        # compute the models score function
        flat_opt_scoreFxn = self.old_policy(flat_states, flat_actions, flat_opt)
        # reshapte this tensor to be time by samples
        opt_scoreFxn = flat_opt_scoreFxn.reshape(reward_tensor.size()).squeeze(2)
        # sum accross time
        sum_opt_scoreFxn = torch.sum(opt_scoreFxn, dim=1)

        """ USE OPTIMALITY VARIABLES AS INDICATORS + COMPUTE LOG LOG P(TAU,O) """
        subopt_reward = torch.log(1 - torch.exp(reward_tensor) + self.epsilon)
        sub_opt_mat = torch.sum((1-optimality_tensor.squeeze(2)) * subopt_reward.squeeze(2), dim=1)
        opt_mat = torch.sum((optimality_tensor.squeeze(2)) * reward_tensor.squeeze(2), dim=1)
        log_joint = sub_opt_mat + opt_mat

        """ USE OPTIMALITY VARIABLES AS INDICATORS + COMPUTE LOG P(O) """
        opt_prob = torch.mv((optimality_tensor.squeeze(2)), torch.log(self.probabilities + self.epsilon) )
        subopt_prob = torch.mv((1.-optimality_tensor.squeeze(2)), torch.log(1 - self.probabilities + self.epsilon))
        logOpt = opt_prob + subopt_prob

        """ COMPUTE BUFFER IMPORTANCE WEIGHTS """
        # sum through time to get the iw
        iw = log_joint.reshape(-1) - sum_opt_scoreFxn.reshape(-1) - logOpt
        # regularize by average sample mean
        iw -= torch.log(total_samples)

        """ CURRENT POLICY SCORE FUNCTION LOG Q_PHI(TAU|O) """
        # compute the models score function
        flat_buffer_opt_scoreFxn = policy(flat_states, flat_actions, flat_opt)
        # reshapte this tensor to be time by samples
        opt_scoreFxn = flat_opt_scoreFxn.reshape(reward_tensor.size()).squeeze(2)
        # sum accross time
        sum_opt_scoreFxn = torch.sum(opt_scoreFxn, dim=1)

        """ COMBINE BUFFER AND NEW UPDATE INFO """
        if self.include_buffer:
            total_iw = torch.cat([iw, buffer_weights])
            total_score = torch.cat([sum_opt_scoreFxn, buffer_sum_opt_scoreFxn])
        else:
            total_iw = iw
            total_score = sum_opt_scoreFxn

        """ CHECK IF WE ARE GOING TO NORMALIZE """
        # if we want to normalize
        if self.normalize == 1:
            # stabalize numerically
            total_iw -= torch.max(total_iw)
            # detach from computation graph
            total_iw = total_iw.detach()
            # compute exponential for the weights
            total_iw = torch.exp(total_iw) / torch.sum(torch.exp(total_iw))
        else:
            # detach from computation graph
            total_iw = iw.detach()
            # compute exponential for the weights
            total_iw = torch.exp(total_iw)

        """ UPDATE BUFFER FOR FUTURE ITERATIONS """
        if self.begin_buffer_updates:
            if self.buffer_update_type == 'sample':
                if self.apply_filtering == 1:
                    self.update_buffer_sample_based(state_tensor, action_tensor, reward_tensor, optimality_tensor, policy, True)
                else:
                    self.update_buffer_sample_based(state_tensor, action_tensor, reward_tensor, optimality_tensor, policy, False)
            elif self.buffer_update_type == 'weight_sample':
                self.update_buffer_weight_sample_based(state_tensor, action_tensor, reward_tensor, optimality_tensor, False, policy, total_iw)
            else:
                self.update_buffer_greedy(state_tensor, action_tensor, reward_tensor, optimality_tensor, iw, policy)

        """ TRUST REGION REGULARIZATION """
        if self.trust_region_reg == 1 and self.KL:
            # keep policy close where it counts
            TRR = self.approx_lagrange * self.KL
        else:
            TRR = torch.tensor(0.)

        """ RETURN THE OBJECTIVE EVALUATION """
        return -1*torch.dot(total_iw, total_score) + TRR.detach(), torch.nonzero(total_iw).size(0), len(total_iw)
