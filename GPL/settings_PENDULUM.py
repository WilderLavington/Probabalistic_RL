
""" THIS FILE CONTAINS THE SETTINGS USED IN THE IWS FOR RL: IN ORDER TO CHANGE
    MAKE ANY CHANGES TO HOW THE ALGORITHM RUNS, THESE PARAMETERS MUST BE
    CHANGED. """

""" BASIC ENVIROMENT INFO  """
# the game enviroment that the agent will interact with
game = 'PENDULUM'
# the total time that the agent will interact with the simulator
trajectory_length = 200
# describes the total samples that are gathered via interaction with the simulator per iteration
sample_size = 25
# total number of training iterations for the algorithm
iterations = 500
# batch size considered at each training update
batch_size = 25
# number of CPU proccesses to handle batches
workers = 1

""" ENVIROMENT SPECIFIC INFORMATION """
# reward shaping
reward_shaping = lambda r: r/2
inv_reward_shaping = lambda r: 2*r
# get state and action dimensions from enviroment
action_size = 1 # [len(env.action_space.sample()) if hasattr(env.action_space.sample(), "__len__") else 1][0]
state_size = 3 #[len(env.reset()) if hasattr(env.reset(), "__len__") else 1][0]
action_box = torch.tensor([-2.0, 2.0])
# must convert to numpy for openai gym
action_transform = lambda action: action.numpy()

""" THIS IS ALL THE INFO ABOUT IMPORTANCE WEIGHTING BASED GPS """
# probability of optimality at each step
optim_prob = 0.99
# whether or not to use normalized importance weighting
normalize = 1
# set adaptive step where we add prob ahead of where agent has solved
set_adaptive = 0
# number of steps ahead of current solve to set to one
adaptive_step = 25
# whether or not to include buffer in training updates
include_buffer = 1
# total size of the buffer used for the update (has to be float)
buffer_size = 50.0
# type up update within buffer
buffer_update_type = 'sample'
# amount of regularization in sampling scheme for sample based buffer updates
sample_reg = 0.001
# include trust region regularization
trust_region_reg = 0
# the approximate lagrange multiplier term to use
approx_lagrange = 0.02

""" THIS IS ALL THE INFO ABOUT THE AGENT MODEL USED """
# size of hidden layer
hidden_layer_size = 128
# this identifies the type of model that will be used to train the agent
agent_model = 'CONTINUOUS'

""" ALL INFO REGARDING THE OPTIMIZATION SCHEME USED """
# this represents the chosen optimization method used
optim = 'Adam'
# this is the stepsize for the chosen optimization method
lr = 5e-4
# this is the weight regularization for the parameters
weight_decay = 0.0
# this is the first order averaging term in Adam
beta_1 = 0.8
# this is the second order averaging term in Adam
beta_2 = 0.9
# this is the learning rate decay in SGD and ASGD
lambd = 1e-4
# this is the averaging term in ASGD
alpha = 0.99
