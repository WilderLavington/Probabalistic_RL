
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
from agent_models import *
from transition_dyna_models import *
from gen_reward_models import *
from rl_variable_imports import *


""" PROGRAM TO CREATE FOLDER AND TEXT FILE TO STORE AND DESCRIBE DATA
    THAT WAS GENERATED DURING AN EXPERIMENT. """
def create_data_storage(game):
    # define directory
    ExperimentID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)])
    path = str(game) + "__" + str(ExperimentID)
    # create directory
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    """ CALL LINUX COMMAND TO STORE INFO IN .TXT"""
    f = os.popen('date \n cat settings_CARTPOLE.py')
    now = f.read()
    file = open(path + "/parameter_info.txt","w")
    file.write(now)
    file.close()

    # add settings text file to directory
    return path

def main():

    """
    PICK TASK:
        CURRENTLY IMPLENTED TASKS INCLUDE,
        1. CARTPOLE
        2. PENDULUM
        3. ACROBOT
        4. MOUNTAIN-CAR DISCRETE
    """

    """ GENERATE A DIRECTORY """
    directory = create_data_storage(game)

    """ INITIALIZE AGENT TRAINING CLASS """
    algorithm = TRAIN_AGENT(game)

    """ SET OPTIM PROBABILITIES """
    optim_probabilities = optim_prob*torch.ones((trajectory_length))

    """ TRAIN AGENT AND GENERATE INFO """
    policy, loss_per_iteration, time_per_iteration, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)

    """ STORE DATA IN THE DIRECTORY """
    torch.save(loss_per_iteration, directory + '/' + 'loss__' +  directory + '.pt')
    torch.save(iw_per_iteration, directory + '/' + 'iw__' + directory + '.pt')
    torch.save(time_per_iteration, directory + '/' + 'time__' + directory + '.pt')


if __name__ == '__main__':
    main()


"""
step 1. initialize policy
        this policy will conform to the following structure:
        a. it will take in a state, and an optimality variable
        b. it will produce an action, as well as the expected reward for that
        action. This will be considered "internal reward" value
step 2. initialize generative model
        this model will produce a sequence of tuples of the the following form:
        a. it will take in the previous state, (this could be the initial state
        or anything else), as well as the action of the agent
        b. it will produce the next state, as well as the associated reward for
        for that state. This reward will be considered the "true reward" values
note 1. the internal reward is trained in such a way as to optimize the agents
        learning and exploration per iteration while the true reward is learned
        to match the rewards produced by interaction with the actual enviroment
step 3. formulation of the model
        the inference model is described in the following way:
        q(tau|O) = pdf(current action, internal rewards | current state, current optimality)
        p(tau,O) = pdf(current state, current action, true reward, current optimality | previous state)

"""
