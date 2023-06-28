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
import torch.optim as optim
from scipy.linalg import svdvals

import cvxpy as cp
from env_single_phase_13bus import IEEE13bus, create_13bus
from safeDDPG import SafePolicyNetwork, ValueNetwork, DDPG

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

A = -np.eye(12)
A[1,0]=1
A[2,0]=1
A[3,0]=1
A[4,1]=1
A[5,2]=1
A[6,2]=1
A[7,2]=1
A[8,3]=1
A[9,5]=1
A[10,5]=1
A[11,7]=1

X = np.diag([0.3856,0.1276,0.3856,0.1119,0.0765,0.0771,0.1928,0.0423,0.1119,0.0766,0.0766,0.0423])
F = -np.linalg.inv(A)
XX =2*F@X@F.T/np.square(4.16)
XX_inv = np.linalg.inv(XX)

net = create_13bus()
injection_bus = np.array([2,7,9])#1,2,5,3,4,7,8,9,6,10,11,12 //2,7,9
env = IEEE13bus(net, injection_bus)

C = np.diag([10000, 0.7*0.15, 10000,10000,10000, 0.5*0.15, 10000, 0.6*0.15, 10000,10000,10000,10000])
Q_limit = np.asarray([[-1.5,1.5],[-1.4,1.4],[-1.0,0.6]])
alpha = 0.5
num_agent = 3
ph_num = 1

seed = 10
torch.manual_seed(seed)
obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100

safe_ddpg_agent_list =[]
index = [1,5,7]
for i in range(3):
    safe_ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device) 
    safe_ddpg_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
                up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[index[i],index[i]],\
                    use_gradient=True, safe_flow=True).to(device)
    safe_ddpg_agent = DDPG(policy_net=safe_ddpg_policy_net, value_net=safe_ddpg_value_net,
                 target_policy_net=safe_ddpg_policy_net, target_value_net=safe_ddpg_value_net) 
    safe_ddpg_agent_list.append(safe_ddpg_agent)
for i in range(3):
    safe_ddpg_policynet_dict = torch.load(f'checkpoints/single-phase/13bus/safe-ddpg/safe-flow_policy_net_checkpoint_a{i}.pth')
    safe_ddpg_agent_list[i].policy_net.load_state_dict(safe_ddpg_policynet_dict)


for i in range(1000):
    state, vpar = env.reset()
    q = np.linalg.inv(C+XX)@(np.ones((12,ph_num))-np.square(vpar.reshape((12,ph_num))))
    state_desired = np.sqrt(XX@q+np.square(vpar.reshape((12,ph_num))))
    if np.any((state_desired - 0.95)<0):
        print('does not hold, too low')
        continue
    elif np.any((state_desired - 1.05)>0):
        print('does not hold, too high')
        continue
    K = np.zeros(12)
    last_action = np.zeros((num_agent,ph_num))
    for step in range(20):
        action = []
        pi = []
        for j in range(num_agent):
            action_agent, pi_agent = safe_ddpg_agent_list[j].policy_net.get_action(np.asarray([state[j]]),last_action[j])
            action.append(action_agent)
            pi.append(pi_agent)
            if np.abs(state[j]-state_desired[index[j]])>0.000001:
                K[index[j]]=action_agent/(state[j]-state_desired[index[j]])
                assert K[index[j]]<0
        if np.linalg.norm(action) < np.linalg.norm(pi):
            print('violate')
        tmp = C@XX_inv+np.eye(12)-np.diag(K)
        test_m = np.transpose(tmp)@tmp - np.diag(K)@np.diag(K)
        eig_v = np.linalg.eigvals(test_m)
        if np.min(eig_v)<0:
            print(np.min(eig_v))
        svdvA = svdvals(C@XX_inv+np.eye(12))
        svdvK = svdvals(np.diag(K)@np.diag(K))
        # if np.min(svdvA)<np.max(svdvK):
        #     print('harsh condition violated')
            
        action = last_action + np.asarray(action)
        next_state, reward, reward_sep, done = env.step_Preward(action, (action-last_action))
        state = next_state
        if done:
            break
        
        