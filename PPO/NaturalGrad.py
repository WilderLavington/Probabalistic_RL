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
from torch.optim.optimizer import Optimizer, required

class VanillaGrad(Optimizer):

    """ NATURAL POLICY GRADIENTS OPTIMIZATION CLASS """
    def __init__(self, params, learning_rate = required, decay = 0.0, L2reg = 0.0, exact = 1):

        """ ERROR CATCHES """
        if learning_rate is not required and learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if L2reg < 0.0:
            raise ValueError("Invalid momentum value: {}".format(L2reg))
        if decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(decay))

        """ DEFAULT INITS"""
        defaults = dict(learning_rate=learning_rate, decay=decay, L2reg=L2reg)

        """ INITIALIZATIONS """
        super(NaturalGrad, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            # set hyper parameters
            learning_rate = group['learning_rate']
            decay = group['decay']
            L2reg = group['L2reg']
            # iterate through parameter groups
            for p in group['params']:

                # get gradient
                d_p = p.grad.data

                # add gradient with specific learning rate to parameter
                p.data.add_(-group['learning_rate'], d_p)

        return None

class NaturalGrad(Optimizer):

    """ NATURAL POLICY GRADIENTS OPTIMIZATION CLASS """
    def __init__(self, params, learning_rate = required, decay = 0.0, L2reg = 0.0, exact = 1):

        """ ERROR CATCHES """
        if learning_rate is not required and learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if L2reg < 0.0:
            raise ValueError("Invalid momentum value: {}".format(L2reg))
        if decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(decay))

        """ DEFAULT INITS"""
        defaults = dict(learning_rate=learning_rate, decay=decay, L2reg=L2reg)

        """ INITIALIZATIONS """
        super(NaturalGrad, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            # set hyper parameters
            learning_rate = group['learning_rate']
            decay = group['decay']
            L2reg = group['L2reg']
            # iterate through parameter groups
            for p in group['params']:

                # get gradient
                d_p = p.grad.data

                # add gradient with specific learning rate to parameter
                p.data.add_(-group['learning_rate'], d_p)

        return None


    def accumulate_scores(self, score_fxn_grads, iw = None):
        if iw == None:
            return FisherInfo
        else:
            return FisherInfo

    # def low_rank_storage(self):
    #     torch.svd()
