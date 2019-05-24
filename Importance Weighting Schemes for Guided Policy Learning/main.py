
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
import random
import string
import sys
import argparse
import numpy as np
import os.path
import json

""" IMPORT PROGRAMS """
from train_agent import *
from game_enviroments import *
from objective_function import *
from train_agent import *
from train_agent_PG import *
from agents import *

""" IMPORT SETTINGS """
from settings import *


""" PROGRAM TO CREATE FOLDER AND TEXT FILE TO STORE AND DESCRIBE DATA
    THAT WAS GENERATED DURING AN EXPERIMENT. """
def create_data_storage():

    return None



def main():

    """
    PICK TASK:
        CURRENTLY IMPLENTED TASKS INCLUDE,
        1. CARTPOLE
        2. PENDULUM
        3. ACROBOT
        4. MOUNTAIN-CAR DISCRETE
    """

    if game == 'CARTPOLE':
        algorithm = CARTPOLE()
        optim_probabilities = optim_prob*torch.ones((args.trajectory_length[0]))
        policy, loss_per_iteration, time_per_iteration, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
        # save it to a tensor
        torch.save(loss_per_iteration, 'loss_tensors/cartpole/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        torch.save(iw_per_iteration, 'loss_tensors/cartpole/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        return None
    if game == 'PENDULUM':
        algorithm = PENDULUM()
        optim_probabilities = optim_prob*torch.ones((args.trajectory_length[0]))
        policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
        # save it to a tensor
        torch.save(loss_per_iteration, 'loss_tensors/cartpole/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        torch.save(iw_per_iteration, 'loss_tensors/cartpole/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        return None
    if game == 'ACROBOT':
        algorithm = ACROBOT()
        optim_probabilities = optim_prob*torch.ones((args.trajectory_length[0]))
        policy, loss_per_iteration, time_per_iteration, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
        # save it to a tensor
        torch.save(loss_per_iteration, 'loss_tensors/cartpole/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        torch.save(iw_per_iteration, 'loss_tensors/cartpole/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        return None
    if game == 'MOUNTAINCAR_DISCRETE':
        algorithm = MOUNTAINCAR_DISCRETE()
        optim_probabilities = optim_prob*torch.ones((args.trajectory_length[0]))
        policy, loss_per_iteration, time_per_iteration, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
        # save it to a tensor
        torch.save(loss_per_iteration, 'loss_tensors/cartpole/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        torch.save(iw_per_iteration, 'loss_tensors/cartpole/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        return None
    else:
        print("please provide a valid game type!!!")
        return None





            # print statements ect
            if args.include_buffer[0] == 1:
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            if args.set_adaptive[0] == 1:
                print("adaptive_step: %r" % args.adaptive_step)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task_adaptive(optim_probabilities, args.adaptive_step[0])
            else:
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)

            return None


    

if __name__ == '__main__':
    main()
