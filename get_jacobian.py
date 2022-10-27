from cv2 import phase
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from env_three_phase_eu import Three_Phase_EU, create_eu_lv
from IEEE_13_3p import IEEE13bus3p, create_13bus3p
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, ReplayBuffer, ReplayBufferPI, PolicyNetwork, SafePolicy3phase, StablePolicy3phase, LinearPolicy


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

seed = 10
torch.manual_seed(seed)
pp_net = create_13bus()

injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
env = IEEE13bus(pp_net, injection_bus)
num_agent = len(injection_bus)
obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100
agent_list = []
# for i in range(num_agent):
#     print(env.network.sgen.at[i, 'bus'])

for i in range(num_agent):    
    policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    agent_list.append(policy_net)

for i in range(num_agent):        
    policynet_dict = torch.load(f'checkpoints/single-phase/13bus/safe-ddpg/policy_net_checkpoint_bus{i}.pth')
    agent_list[i].load_state_dict(policynet_dict)
    # agent_list[i] = agent_list[i].double()
for i in range(num_agent):
    # plot policy
    N = 400
    jax_array= torch.zeros((N,1))
    
    for j in range(N):
        state = np.array([0.8+0.001*j])
        
        x = torch.tensor(state).to(device)
        x.requires_grad = True      
        
        # action_jb = agent_list[i](x)
    #     output = torch.ones(1).to(device)
    #     jacX=torch.autograd.grad(action_jb, x, grad_outputs=output, retain_graph=True)[0]    
        
        jacX = torch.autograd.functional.jacobian(agent_list[i],x)  
        jax_array[j]=jacX
    print(torch.max(jax_array))
        


    