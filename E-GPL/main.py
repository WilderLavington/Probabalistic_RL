
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
from agents import *
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

    """ TRAIN AGENT AND GENERATE INFO """
    policy, loss_per_iteration, time_per_iteration, iw_per_iteration = algorithm.train_gym_task()

    """ STORE DATA IN THE DIRECTORY """
    torch.save(loss_per_iteration, directory + '/' + 'loss__' +  directory + '.pt')
    torch.save(iw_per_iteration, directory + '/' + 'iw__' + directory + '.pt')
    torch.save(time_per_iteration, directory + '/' + 'time__' + directory + '.pt')


if __name__ == '__main__':
    main()
