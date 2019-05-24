

""" THIS FILE CONTAINS THE SETTINGS USED IN THE IWS FOR RL: IN ORDER TO CHANGE
    MAKE ANY CHANGES TO HOW THE ALGORITHM RUNS, THESE PARAMETERS MUST BE
    CHANGED. """

# the game enviroment that the agent will interact with
game = 'CARTPOLE'
# the total time that the agent will interact with the simulator
trajectory_length = 200
# this identifies the type of model that will be used to train the agent
agent_model = 'DISCRETE'
# describes the total samples that are gathered via interaction with the simulator per iteration
sample_size = 25
# total number of training iterations for the algorithm
iterations = 500
# batch size considered at each training update
batch_size = 25
# number of CPU proccesses to handle batches
workers = 1
# probability of optimality at each step
optim_prob = 0.99
# whether or not to use normalized importance weighting
normalize = 1



# adaptive info
CLI.add_argument("--set_adaptive", nargs="*",  type=int, default=[0])
CLI.add_argument("--adaptive_step", nargs="*",  type=int, default=[25])

# buffer and filter info
CLI.add_argument("--include_buffer", nargs="*",  type=int, default=[0])
CLI.add_argument("--buffer_size", nargs="*",  type=int, default=[100])
CLI.add_argument("--buffer_update_type", nargs="*", type=str, default=['sample'])
CLI.add_argument("--sample_reg", nargs="*",  type=int, default=[0.])
CLI.add_argument("--apply_filtering", nargs="*",  type=int, default=[0])

# trust region info
CLI.add_argument("--trust_region_reg", nargs="*",  type=int, default=[0])
CLI.add_argument("--approx_lagrange", nargs="*",  type=float, default=[0.01])

# PG params (objective types)
CLI.add_argument("--max_ent_reg", nargs="*", type=int, default=[0])
CLI.add_argument("--max_ent_reg_decay", nargs="*", type=int, default=[0.])

# Trust region methods params
CLI.add_argument("--PPO", nargs="*", type=int, default=[0])
CLI.add_argument("--TRPO", nargs="*", type=int, default=[0])

# advantage estimation
CLI.add_argument("--GAE", nargs="*", type=int, default=[0])

# off policy sampling
CLI.add_argument("--off_policy_samples", nargs="*", type=int, default=[0.])
CLI.add_argument("--off_policy_batch_size", nargs="*", type=int, default=[0.])

# AC params
CLI.add_argument("--AC", nargs="*", type=int, default=[0])
CLI.add_argument("--A2C", nargs="*", type=int, default=[0])
CLI.add_argument("--A3C", nargs="*", type=int, default=[0])

# Q learning params

# optimization params
CLI.add_argument("--optim", nargs="*",  type=str, default=['Adam'])
CLI.add_argument("--lr", nargs="*",  type=float, default=[5e-4])
CLI.add_argument("--weight_decay", nargs="*",  type=float, default=[0.0])
CLI.add_argument("--beta_1", nargs="*",  type=float, default=[0.8])
CLI.add_argument("--beta_2", nargs="*",  type=float, default=[0.9])
CLI.add_argument("--lambd", nargs="*",  type=float, default=[1e-2])
CLI.add_argument("--alpha", nargs="*",  type=float, default=[0.75])
