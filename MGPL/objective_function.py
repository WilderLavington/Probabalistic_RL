
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

class PHI_WAKE(torch.nn.Module):

    """
        REWEIGHTED WAKE SLEEP ALGORITHM: INFERENCE NETWORK WAKE PHASE.
        IN THIS STEP, EXACTLY THE SAME AS THE MODEL FREE CASE, WE TAKE
        THE TRAJECTORIES GENERATED UNDER THE POLICY AND SIMULATOR AND THEN
        TRAIN THE AGENT FOLLOWING THE IMPORTANCE WEIGHTING SCHEME DESCRIBED
        THEREIN.
    """

    def __init__(self, trajectory_length, simulations, probabilities, normalize, rev_reward_shaping, buffer_update_type, \
                sample_reg, trust_region_reg, approx_lagrange, use_running_avg, running_avg_norm, running_avg_count):
        """ INITIALIZATIONS """
        super(PHI_WAKE, self).__init__()
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.probabilities = probabilities
        self.epsilon = 0.0000001
        # simple normalized importance sampling
        self.normalize = normalize
        # if we want to normalize using an exponential moving average
        self.use_running_avg = use_running_avg
        self.running_avg_norm = running_avg_norm
        self.running_avg_count = running_avg_count
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
        self.buffer_size = int(buffer_size)
        return None

    def start_buffer_updates(self, use):
        self.begin_buffer_updates = use
        return None

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
            samples = torch.stack([buffer_distribution.sample() for i in range(self.buffer_size)])
            # store info
            self.buffer_states = new_states[samples,:,:]
            self.buffer_action = new_actions[samples,:,:]
            self.buffer_reward = new_rewards[samples,:,:]
            self.buffer_optim = new_optim[samples,:]
            self.buffer_iw = iw[samples]
            self.buffer_set = True
            self.current_buffer_size = self.buffer_size
        else:
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
            samples = torch.stack([buffer_distribution.sample() for i in range(self.buffer_size)])
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

        """ RETURN """
        return self.buffer_states, self.buffer_action, self.buffer_reward, self.buffer_optim, buffer_distribution

    """ LOSS FUNCTION FOR REINFORCEMENT LEARNING AS IMPORTANCE WEIGHTED APPROXIMATE INFERENCE FOLLOWING KL(P||Q) """
    def forward(self, policy, state_tensor, action_tensor, reward_tensor, optimality_tensor):

        # compute total number of samples used in the update
        """ COMPUTE ADDITION OF BUFFER INFO """
        total_samples = torch.tensor(1.0*reward_tensor.size()[0])
        if self.old_policy == None:
            self.old_policy = policy

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
            buffer_sum_opt_scoreFxn = torch.log(torch.exp(buffer_sum_opt_scoreFxn) + 1 / (self.buffer_size))

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

        """ CHECK IF WE ARE GOING TO NORMALIZE IN SOME WAY """
        # if we want to normalize
        if self.normalize == 1:
                # stabalize numerically
                total_iw -= torch.max(total_iw)
                # detach from computation graph
                total_iw = total_iw.detach()
                # compute exponential for the weights
                total_iw = torch.exp(total_iw) / torch.sum(torch.exp(total_iw))
        # if we want to use a running average to numerically stabalize
        elif self.use_running_avg:
            # compute exponential moving average
            self.running_avg_norm = self.running_avg_norm + (torch.max(total_iw) - self.running_avg_norm) / self.running_avg_count
            # stabalize numerically
            total_iw -= self.running_avg_norm
            # detach from computation graph
            total_iw = total_iw.detach()
            # compute exponential for the weights
            total_iw = torch.exp(total_iw)
            self.running_avg_count += 1
        else:
            # detach from computation graph
            total_iw = iw.detach()
            # compute exponential for the weights
            total_iw = torch.exp(total_iw)

        """ CHECK WE ARE USING THE EFFECTIVE SAMPLE SIZE """
        n_e = torch.sum(total_iw) ** 2 / torch.sum(total_iw ** 2)
        # if not increase sample size by 1 upto 750
        if self.buffer_size:
            print("Effective sample size maintained: " + str(n_e.numpy() < len(total_iw) and 1 < n_e.numpy()))
            print("Current buffer size: " + str(self.buffer_size))
            if (n_e.numpy() < len(total_iw) and 1 < n_e.numpy()):
                if self.buffer_size > 50:
                    self.buffer_size -= 1
            else:
                if int(self.buffer_size) <= 725:
                    self.buffer_size += 25

        """ UPDATE BUFFER FOR FUTURE ITERATIONS """
        if self.begin_buffer_updates:
            self.update_buffer_IWS(state_tensor, action_tensor, reward_tensor, optimality_tensor, False, policy, total_iw)

        """ TRUST REGION REGULARIZATION """
        if self.trust_region_reg == 1:
            # keep policy close where it counts
            TRR = self.approx_lagrange * self.KL
        else:
            TRR = torch.tensor(0.)

        """ UPDATE OLD POLICY """
        self.old_policy = policy

        """ RETURN THE OBJECTIVE EVALUATION """
        return -1*torch.dot(total_iw, total_score) + TRR.detach(), torch.nonzero(total_iw).size(0), len(total_iw)

class PHI_SLEEP(torch.nn.Module):

    """
        REWEIGHTED WAKE SLEEP ALGORITHM: INFERENCE NETWORK SLEEP PHASE.
        IN THIS PART OF THE ALGORITHM, THE AGENT TAKES SAMPLES GENERATED
        UNDER THE GENERATIVE MODEL AND TRAINS ON THEM. FOR NOW, THIS
        PROGRAM IS THE EXACT SAME AS THE PHI WAKE PHASE HOWEVER I CREATED
        TWO PROGRAMS SO THAT IN THE FUTURE I COULD MAKE MODIFICATIONS.
    """

    def __init__(self, trajectory_length, simulations, probabilities, normalize, rev_reward_shaping, buffer_update_type, \
                sample_reg, trust_region_reg, approx_lagrange, use_running_avg, running_avg_norm, running_avg_count):
        """ INITIALIZATIONS """
        super(PHI_SLEEP, self).__init__()
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.probabilities = probabilities
        self.action_size = 2
        self.epsilon = 0.0000001
        # simple normalized importance sampling
        self.normalize = normalize
        # if we want to normalize using an exponential moving average
        self.use_running_avg = use_running_avg
        self.running_avg_norm = running_avg_norm
        self.running_avg_count = running_avg_count
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
        self.buffer_size = int(buffer_size)
        return None

    def start_buffer_updates(self, use):
        self.begin_buffer_updates = use
        return None

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
            samples = torch.stack([buffer_distribution.sample() for i in range(self.buffer_size)])
            # store info
            self.buffer_states = new_states[samples,:,:]
            self.buffer_action = new_actions[samples,:,:]
            self.buffer_reward = new_rewards[samples,:,:]
            self.buffer_optim = new_optim[samples,:]
            self.buffer_iw = iw[samples]
            self.buffer_set = True
            self.current_buffer_size = self.buffer_size
        else:
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
            samples = torch.stack([buffer_distribution.sample() for i in range(self.buffer_size)])
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

        """ RETURN """
        return self.buffer_states, self.buffer_action, self.buffer_reward, self.buffer_optim, buffer_distribution

    """ LOSS FUNCTION FOR REINFORCEMENT LEARNING AS IMPORTANCE WEIGHTED APPROXIMATE INFERENCE FOLLOWING KL(P||Q) """
    def forward(self, policy, state_tensor, action_tensor, reward_tensor, optimality_tensor):

        # compute total number of samples used in the update
        """ COMPUTE ADDITION OF BUFFER INFO """
        total_samples = torch.tensor(1.0*reward_tensor.size()[0])
        if self.old_policy == None:
            self.old_policy = policy

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
            buffer_sum_opt_scoreFxn = torch.log(torch.exp(buffer_sum_opt_scoreFxn) + 1 / (self.buffer_size))

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

        """ CHECK IF WE ARE GOING TO NORMALIZE IN SOME WAY """
        # if we want to normalize
        if self.normalize == 1:
                # stabalize numerically
                total_iw -= torch.max(total_iw)
                # detach from computation graph
                total_iw = total_iw.detach()
                # compute exponential for the weights
                total_iw = torch.exp(total_iw) / torch.sum(torch.exp(total_iw))
        # if we want to use a running average to numerically stabalize
        elif self.use_running_avg:
            # compute exponential moving average
            self.running_avg_norm = self.running_avg_norm + (torch.max(total_iw) - self.running_avg_norm) / self.running_avg_count
            # stabalize numerically
            total_iw -= self.running_avg_norm
            # detach from computation graph
            total_iw = total_iw.detach()
            # compute exponential for the weights
            total_iw = torch.exp(total_iw)
            self.running_avg_count += 1
        else:
            # detach from computation graph
            total_iw = iw.detach()
            # compute exponential for the weights
            total_iw = torch.exp(total_iw)

        """ CHECK WE ARE USING THE EFFECTIVE SAMPLE SIZE """
        n_e = torch.sum(total_iw) ** 2 / torch.sum(total_iw ** 2)
        # if not increase sample size by 1 upto 750
        if self.buffer_size:
            print("Effective sample size maintained: " + str(n_e.numpy() < len(total_iw) and 1 < n_e.numpy()))
            print("Current buffer size: " + str(self.buffer_size))
            if (n_e.numpy() < len(total_iw) and 1 < n_e.numpy()):
                if self.buffer_size > 50:
                    self.buffer_size -= 1
            else:
                if int(self.buffer_size) <= 725:
                    self.buffer_size += 25

        """ UPDATE BUFFER FOR FUTURE ITERATIONS """
        if self.begin_buffer_updates:
            self.update_buffer_IWS(state_tensor, action_tensor, reward_tensor, optimality_tensor, False, policy, total_iw)

        """ TRUST REGION REGULARIZATION """
        if self.trust_region_reg == 1:
            # keep policy close where it counts
            TRR = self.approx_lagrange * self.KL
        else:
            TRR = torch.tensor(0.)

        """ UPDATE OLD POLICY """
        self.old_policy = policy

        """ RETURN THE OBJECTIVE EVALUATION """
        return -1*torch.dot(total_iw, total_score) + TRR.detach(), torch.nonzero(total_iw).size(0), len(total_iw)

class THETA_WAKE(torch.nn.Module):

    """
        REWEIGHTED WAKE SLEEP ALGORITHM: GENERATIVE NETWORK WAKE PHASE.
        IN THIS STAGE WE STEP THE GENERATIVE MODELS OF THE ENVIRMENT
        DYNAMICS AND REWARD MODELS. 
    """
    def __init__(self, trajectory_length, simulations, probabilities, normalize, rev_reward_shaping):
        """ INITIALIZATIONS """
        super(THETA_WAKE_REWARD, self).__init__()
        # general init info
        self.simulations = simulations
        self.trajectory_length = trajectory_length
        self.probabilities = probabilities
        self.action_size = 2
        self.epsilon = 0.0000001
        # simple normalized importance sampling
        self.normalize = normalize

    def forward(self, policy, state_tensor, action_tensor, reward_tensor, optimality_tensor):

        """ OLD POLICY SCORE FUNCTION LOG Q_OLD(TAU|O) """
        # convert format to something we can feed to model
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)
        flat_opt = torch.flatten(optimality_tensor, start_dim=0,end_dim=1)
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

        """ CHECK IF WE ARE GOING TO NORMALIZE IN SOME WAY """
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

        """ CHECK WE ARE USING THE EFFECTIVE SAMPLE SIZE """
        n_e = torch.sum(total_iw) ** 2 / torch.sum(total_iw ** 2)
        # if not increase sample size by 1 upto 750
        if self.buffer_size:
            print("Effective sample size maintained: " + str(n_e.numpy() < len(total_iw) and 1 < n_e.numpy()))
            print("Current buffer size: " + str(self.buffer_size))
            if (n_e.numpy() < len(total_iw) and 1 < n_e.numpy()):
                if self.buffer_size > 50:
                    self.buffer_size -= 1
            else:
                if int(self.buffer_size) <= 725:
                    self.buffer_size += 25

        """ UPDATE BUFFER FOR FUTURE ITERATIONS """
        if self.begin_buffer_updates:
            self.update_buffer_IWS(state_tensor, action_tensor, reward_tensor, optimality_tensor, False, policy, total_iw)

        """ TRUST REGION REGULARIZATION """
        if self.trust_region_reg == 1:
            # keep policy close where it counts
            TRR = self.approx_lagrange * self.KL
        else:
            TRR = torch.tensor(0.)

        """ UPDATE OLD POLICY """
        self.old_policy = policy

        """ RETURN THE OBJECTIVE EVALUATION """
        return -1*torch.dot(total_iw, total_score) + TRR.detach(), torch.nonzero(total_iw).size(0), len(total_iw)
