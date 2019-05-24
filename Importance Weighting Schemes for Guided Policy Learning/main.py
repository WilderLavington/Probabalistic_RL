
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

def main():
    

    # parse command line arguments
    args = CLI.parse_args()

    # print general training info
    print("sample_size: %r" % args.sample_size)
    print("iterations: %r" % args.iterations)
    print("batch_size: %r" % args.batch_size)
    print("workers: %r" % args.workers)
    print("trajectory_length: %r" % args.trajectory_length)

    # print optim params
    print("Optimizer: %r" % args.optim)
    if args.optim[0] == 'Adam':
        print("lr: %r" % args.lr)
        print("weight_decay: %r" % args.weight_decay)
        print("beta_1: %r" % args.beta_1)
        print("beta_2: %r" % args.beta_2)
        # set adam params
        params = {"lr": args.lr[0], "betas": (args.beta_1[0], args.beta_2[0]), "weight_decay": args.weight_decay[0]}
    elif args.optim[0] == 'ASGD':
        print("lr: %r" % args.lr)
        print("weight_decay: %r" % args.weight_decay)
        print("lambd: %r" % args.lambd)
        print("alpha: %r" % args.alpha)
        # set ASGD params
        params = {"lr": args.lr[0], "lambd": args.lambd[0], "alpha": args.alpha[0], "weight_decay": args.weight_decay[0]}
    elif args.optim[0] == 'SGD':
        print("lr: %r" % args.lr)
        print("weight_decay: %r" % args.weight_decay)
        print("lambd: %r" % args.lambd)
        print("alpha: %r" % args.alpha)
        # set SGD params
        params = {"lr": args.lr[0], "lambd": args.lambd[0], "alpha": args.alpha[0], "weight_decay": args.weight_decay[0]}
    elif args.optim[0] == 'NaturalGrad':
        print("lr: %r" % args.lr)
        print("lambd: %r" % args.lambd)
        print("weight_decay: %r" % args.weight_decay)
        # set Natural gradient params
        params = {"lr": args.lr[0], "lambd": args.lambd[0], "weight_decay": args.weight_decay[0]}

    # evaluation
    """ PICK TASK """
    print("game: %r" % args.game)
    if args.game[0] == 'CARTPOLE':
        """ PICK OBJECTIVE """
        print("objective: %r" % args.objective)
        if args.objective[0] == 'RWS':
            print("include_buffer: %r" % args.include_buffer)
            print("normalize: %r" % args.normalize)
            print("optim_prob: %r" % args.optim_prob)
            print("set_adaptive: %r" % args.set_adaptive)
            print("trust_region_reg: %r" % args.trust_region_reg)
            if args.trust_region_reg[0] == 1:
                print("approx_lagrange: %r" % args.approx_lagrange)
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = CARTPOLE('CartPole-v0', args.iterations[0], args.sample_size[0], args.trajectory_length[0], args.batch_size[0], \
                            args.workers[0], args.normalize[0], params, args.include_buffer[0], args.buffer_size[0], args.optim[0], \
                            args.buffer_update_type[0], args.sample_reg[0],  args.apply_filtering[0], args.trust_region_reg[0], \
                            args.approx_lagrange[0])
            # set optimality
            optim_probabilities = args.optim_prob[0]*torch.ones((args.trajectory_length[0]))
            # print statements ect
            if args.include_buffer[0] == 1:
                print("buffer_size: %r" % args.buffer_size)
                print("buffer_update_type: %r" % args.buffer_update_type)
                print("apply_filtering: %r" % args.apply_filtering)
                print("sample_reg: %r" % args.sample_reg)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            if args.set_adaptive[0] == 1:
                print("adaptive_step: %r" % args.adaptive_step)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task_adaptive(optim_probabilities, args.adaptive_step[0])
            else:
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/cartpole/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            torch.save(iw_per_iteration, 'loss_tensors/cartpole/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            return None
        elif args.objective[0] == 'PG':
            print("max_ent_reg: %r" % args.max_ent_reg)
            print("max_ent_reg_decay: %r" % args.max_ent_reg_decay)
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = PG_CARTPOLE('CartPole-v0', args.iterations[0], args.sample_size[0], args.trajectory_length[0], \
                args.batch_size[0], args.workers[0], adam_params)
            # train algorithm
            policy, loss_per_iteration, _ = algorithm.train_gym_task()
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/cartpolePG/loss_replay_SO_' + str(ID) + '.pt')
            return None
        elif args.objective[0] == 'PPO':
            return None
        elif args.objective[0] == 'TRPO':
            return None
        elif args.objective[0] == 'SAC':
            return None
        elif args.objective[0] == 'A3C':
            return None
        elif args.objective[0] == 'A2C':
            return None
        elif args.objective[0] == 'TD':
            return None
        else:
            print("error mah brother")
            return None
    elif args.game[0] == 'PENDULUM':
        """ PICK OBJECTIVE """
        if args.objective[0] == 'RWS':
            print("include_buffer: %r" % args.include_buffer)
            print("normalize: %r" % args.normalize)
            print("optim_prob: %r" % args.optim_prob)
            print("set_adaptive: %r" % args.set_adaptive)
            print("trust_region_reg: %r" % args.trust_region_reg)
            if args.trust_region_reg[0] == 1:
                print("approx_lagrange: %r" % args.approx_lagrange)
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = PENDULUM('Pendulum-v0', args.iterations[0], args.sample_size[0], args.trajectory_length[0], args.batch_size[0], \
                            args.workers[0], args.normalize[0], params, args.include_buffer[0], args.buffer_size[0], args.optim[0], \
                            args.buffer_update_type[0], args.sample_reg[0],  args.apply_filtering[0], args.trust_region_reg[0], \
                            args.approx_lagrange[0])
            # set optimality
            optim_probabilities = args.optim_prob[0]*torch.ones((args.trajectory_length[0]))
            # print statements ect
            if args.include_buffer[0] == 1:
                print("buffer_size: %r" % args.buffer_size)
                print("buffer_update_type: %r" % args.buffer_update_type)
                print("apply_filtering: %r" % args.apply_filtering)
                print("sample_reg: %r" % args.sample_reg)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            if args.set_adaptive[0] == 1:
                print("adaptive_step: %r" % args.adaptive_step)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task_adaptive(optim_probabilities, args.adaptive_step[0])
            else:
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/pendulum/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            torch.save(iw_per_iteration, 'loss_tensors/pendulum/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
        elif args.objective[0] == 'PG':
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = PG_PENDULUM('Pendulum-v0', args.iterations[0], args.sample_size[0], args.trajectory_length[0], \
                args.batch_size[0], args.workers[0], adam_params)
            # train algorithm
            policy, loss_per_iteration, _ = algorithm.train_gym_task()
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/pendulumPG/loss_replay_SO_' + str(ID) + '.pt')
            return None
        elif args.objective[0] == 'PPO':
            return None
        elif args.objective[0] == 'TRPO':
            return None
        elif args.objective[0] == 'SAC':
            return None
        elif args.objective[0] == 'A3C':
            return None
        elif args.objective[0] == 'A2C':
            return None
        elif args.objective[0] == 'TD':
            return None
        else:
            print("error mah brother")
            return None
    elif args.game[0] == 'MOUNTAINCAR':
        if args.objective[0] == 'RWS':
            print("include_buffer: %r" % args.include_buffer)
            print("normalize: %r" % args.normalize)
            print("optim_prob: %r" % args.optim_prob)
            print("set_adaptive: %r" % args.set_adaptive)
            print("trust_region_reg: %r" % args.trust_region_reg)
            if args.trust_region_reg[0] == 1:
                print("approx_lagrange: %r" % args.approx_lagrange)
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = MOUNTAINCAR('MountainCar-v0', args.iterations[0], args.sample_size[0], args.trajectory_length[0], args.batch_size[0], \
                            args.workers[0], args.normalize[0], params, args.include_buffer[0], args.buffer_size[0], args.optim[0], \
                            args.buffer_update_type[0], args.sample_reg[0],  args.apply_filtering[0], args.trust_region_reg[0], \
                            args.approx_lagrange[0])
            # set optimality
            optim_probabilities = args.optim_prob[0]*torch.ones((args.trajectory_length[0]))
            # print statements ect
            if args.include_buffer[0] == 1:
                print("buffer_size: %r" % args.buffer_size)
                print("buffer_update_type: %r" % args.buffer_update_type)
                print("apply_filtering: %r" % args.apply_filtering)
                print("sample_reg: %r" % args.sample_reg)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            if args.set_adaptive[0] == 1:
                print("adaptive_step: %r" % args.adaptive_step)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task_adaptive(optim_probabilities, args.adaptive_step[0])
            else:
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/mountaincar/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            torch.save(iw_per_iteration, 'loss_tensors/mountaincar/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            return None
        elif args.objective[0] == 'PG':
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = PG_MOUNTAINCAR('MountainCar-v0', args.iterations[0], args.sample_size[0], args.trajectory_length[0], args.batch_size[0], args.workers[0])
            # train algorithm
            policy, loss_per_iteration, _ = algorithm.train_gym_task()
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/mountaincarPG/loss_replay_SO_' + str(ID) + '.pt')
            return None
        else:
            print("error mah brother")
            return None
    elif args.game[0] == 'ACROBOT':
        """ PICK OBJECTIVE """
        if args.objective[0] == 'RWS':
            print("include_buffer: %r" % args.include_buffer)
            print("normalize: %r" % args.normalize)
            print("optim_prob: %r" % args.optim_prob)
            print("set_adaptive: %r" % args.set_adaptive)
            print("trust_region_reg: %r" % args.trust_region_reg)
            if args.trust_region_reg[0] == 1:
                print("approx_lagrange: %r" % args.approx_lagrange)
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = ACROBOT('Acrobot-v1', args.iterations[0], args.sample_size[0], args.trajectory_length[0], args.batch_size[0], \
                            args.workers[0], args.normalize[0], params, args.include_buffer[0], args.buffer_size[0], args.optim[0], \
                            args.buffer_update_type[0], args.sample_reg[0], args.apply_filtering[0], args.trust_region_reg[0], \
                            args.approx_lagrange[0])
            # set optimality
            optim_probabilities = args.optim_prob[0]*torch.ones((args.trajectory_length[0]))
            # print statements ect
            if args.include_buffer[0] == 1:
                print("buffer_size: %r" % args.buffer_size)
                print("buffer_update_type: %r" % args.buffer_update_type)
                print("apply_filtering: %r" % args.apply_filtering)
                print("sample_reg: %r" % args.sample_reg)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            if args.set_adaptive[0] == 1:
                print("adaptive_step: %r" % args.adaptive_step)
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task_adaptive(optim_probabilities, args.adaptive_step[0])
            else:
                policy, loss_per_iteration, _, iw_per_iteration = algorithm.train_gym_task(optim_probabilities)
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/acrobot/loss_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            torch.save(iw_per_iteration, 'loss_tensors/acrobot/iw_replay_SO' + str(ID) + '_with_'+ str(args.optim_prob) + '.pt')
            return None
        elif args.objective[0] == 'PG':
            # generate an ID for the task
            ID = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(12)])
            # set algorithm
            algorithm = PG_ACROBOT('Acrobot-v1', args.iterations[0], args.sample_size[0], args.trajectory_length[0], args.batch_size[0], args.workers[0])
            # train algorithm
            policy, loss_per_iteration, _ = algorithm.train_gym_task()
            # save it to a tensor
            torch.save(loss_per_iteration, 'loss_tensors/acrobotPG/loss_replay_SO_' + str(ID) + '.pt')
            return None
        elif args.objective[0] == 'PPO':
            return None
        elif args.objective[0] == 'TRPO':
            return None
        elif args.objective[0] == 'SAC':
            return None
        elif args.objective[0] == 'A3C':
            return None
        elif args.objective[0] == 'A2C':
            return None
        elif args.objective[0] == 'TD':
            return None
        else:
            print("error mah brother")
            return None

    else:
        print("please include a vaible game")

if __name__ == '__main__':
    main()
