### import collections
from cProfile import label
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

from environment_single_phase import create_56bus, VoltageCtrl_nonlinear
from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, ReplayBuffer, ReplayBufferPI, PolicyNetwork, SafePolicy3phase
from IEEE_13_3p import IEEE13bus3p, create_13bus3p

from scipy.io import loadmat

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

PV_p = loadmat('PV.mat')
p = loadmat('aggr_p.mat')
q = loadmat('aggr_q.mat')


pp_net = create_13bus()
injection_bus = np.array([2, 7, 9])
injection_bus = np.array([1,2,3,4,5,6, 7, 8,9,10,11,12])
env = IEEE13bus(pp_net, injection_bus)
num_agent = len(injection_bus)

max_ac = 0.3
ph_num = 1
slope = 2
seed = 10
torch.manual_seed(seed)
plt.rcParams['font.size'] = '15'

if ph_num == 3:
    type_name = 'three-phase'
else:
    type_name = 'single-phase'

obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100


ddpg_agent_list = []
safe_ddpg_agent_list = []

for i in range(num_agent):
    safe_ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    
    safe_ddpg_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    
    # ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)  
    # ddpg_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    

    # ddpg_agent = DDPG(policy_net=ddpg_policy_net, value_net=ddpg_value_net,
    #              target_policy_net=ddpg_policy_net, target_value_net=ddpg_value_net)
    
    safe_ddpg_agent = DDPG(policy_net=safe_ddpg_policy_net, value_net=safe_ddpg_value_net,
                 target_policy_net=safe_ddpg_policy_net, target_value_net=safe_ddpg_value_net)    
    
    # ddpg_agent_list.append(ddpg_agent)
    safe_ddpg_agent_list.append(safe_ddpg_agent)

for i in range(num_agent):
    # ddpg_valuenet_dict = torch.load(f'checkpoints/{type_name}/13bus/ddpg/value_net_checkpoint_a{i}.pth')
    # ddpg_policynet_dict = torch.load(f'checkpoints/{type_name}/13bus/ddpg/policy_net_checkpoint_a{i}.pth')
    # ddpg_agent_list[i].value_net.load_state_dict(ddpg_valuenet_dict)
    # ddpg_agent_list[i].policy_net.load_state_dict(ddpg_policynet_dict)

    safe_ddpg_valuenet_dict = torch.load(f'checkpoints/{type_name}/13bus/safe-ddpg/value_net_checkpoint_bus{i}.pth')
    safe_ddpg_policynet_dict = torch.load(f'checkpoints/{type_name}/13bus/safe-ddpg/policy_net_checkpoint_bus{i}.pth')
    safe_ddpg_agent_list[i].value_net.load_state_dict(safe_ddpg_valuenet_dict)
    safe_ddpg_agent_list[i].policy_net.load_state_dict(safe_ddpg_policynet_dict)

def plot_traj(p,q,pv_p):
    ddpg_plt=[]
    safe_plt = []
    ddpg_a_plt=[]
    safe_a_plt = []

    state = env.reset0()
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    
    
    for step in range(p.shape[0]):
        action = np.zeros((num_agent,1))
        next_state, reward, reward_sep, done = env.step_load(action, p[step],q[step],pv_p[step])
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    fig, axs = plt.subplots(1, 3, figsize=(12,3.5))
    plt.gcf().subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.18)
    axs[0].plot(range(len(action_list)), p, label = f'Active Load', linewidth=1.5)
    axs[0].plot(range(len(action_list)), q, label = f'Reactive Load', linewidth=1.5)
    axs[0].plot(range(len(action_list)), pv_p, label = f'Solar', linewidth=1.5)
    
    # np.save('ddpg_list.npy',state_list)
    # with open('ddpg_list.npy', 'rb') as f:
    #     state_list = np.load(f)
    for i in range(num_agent):    
        dps = axs[1].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], label = f'Bus {injection_bus[i]}', linewidth=1.5)
    
    axs[1].plot(range(len(action_list)), [0.95]*len(action_list), '--', color='k', linewidth=1)
    axs[1].plot(range(len(action_list)), [1.05]*len(action_list), '--', color='k', linewidth=1)

    state = env.reset0()
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(p.shape[0]):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))#+np.random.normal(0, 0.05)
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_load(action, p[step],q[step],pv_p[step])
    
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    # safe_name = []
    # np.save('sddpg_list.npy',state_list)
    # with open('sddpg_list.npy', 'rb') as f:
    #     state_list = np.load(f)
    for i in range(num_agent):    
        safes=axs[2].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], label = f'Bus {injection_bus[i]}', linewidth=1.5)
    axs[2].plot(range(len(action_list)), [0.95]*len(action_list), '--', color='k', linewidth=1)
    axs[2].plot(range(len(action_list)), [1.05]*len(action_list), '--', color='k', linewidth=1)
    # leg1 = plt.legend(safe_a_plt, safe_name, loc='lower left')
    axs[0].legend(loc='upper left', prop={"size":10})
    box = axs[1].get_position()
    axs[1].set_position([box.x0, box.y0,
                 box.width, box.height * 0.8])
    box = axs[2].get_position()
    axs[2].set_position([box.x0, box.y0,
                 box.width, box.height * 0.8])
    axs[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.4),
          fancybox=True, shadow=True, ncol=6, prop={"size":10})
    # axs[2].legend(loc='lower right', prop={"size":10},ncol=2)
    axs[0].set_xlabel('Time (Hour)')   
    axs[1].set_xlabel('Time (Hour)')  
    axs[2].set_xlabel('Time (Hour)')  
    # axs[2].get_yaxis().set_visible(False)
    axs[1].set_yticks([0.95,1.00,1.05,1.10])
    axs[1].set_yticklabels(['0.95','1.00','1.05','1.10'])
    axs[2].set_yticks([0.95,1.00,1.05,1.10])
    axs[2].set_yticklabels(['0.95','1.00','1.05','1.10'])
    axs[0].set_xticks(np.arange(0,len(action_list),3600))
    axs[0].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
    axs[1].set_xticks(np.arange(0,len(action_list),3600))
    axs[1].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
    axs[2].set_xticks(np.arange(0,len(action_list),3600))
    axs[2].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
    axs[0].set_ylabel('Power (MW/MVar)', fontsize=15)   
    axs[1].set_ylabel('Bus voltage (p.u.)', fontsize=15)  
    axs[2].set_ylabel('Bus voltage (p.u.)', fontsize=15)  
    plt.show()

def plot_traj_no_leg(p,q,pv_p):
    ddpg_plt=[]
    safe_plt = []
    ddpg_a_plt=[]
    safe_a_plt = []

    state = env.reset0()
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    
    
    for step in range(p.shape[0]):
        action = np.zeros((num_agent,1))
        next_state, reward, reward_sep, done = env.step_load(action, p[step],q[step],pv_p[step])
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    plt.gcf().subplots_adjust(wspace=0.4)
    plt.gcf().subplots_adjust(bottom=0.18)
    axs[0].plot(range(len(action_list)), p[:len(action_list)], label = f'Active Load', linewidth=1.5)
    axs[0].plot(range(len(action_list)), q[:len(action_list)], label = f'Reactive Load', linewidth=1.5)
    axs[0].plot(range(len(action_list)), pv_p[:len(action_list)], label = f'Solar', linewidth=1.5)
    
    # np.save('ddpg_list.npy',state_list)
    # with open('ddpg_list.npy', 'rb') as f:
    #     state_list = np.load(f)
    for i in range(num_agent):    
        dps = axs[1].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], label = f'Bus {injection_bus[i]}', linewidth=1.5)
    
    axs[1].plot(range(len(action_list)), [0.95]*len(action_list), '--', color='k', linewidth=1)
    axs[1].plot(range(len(action_list)), [1.05]*len(action_list), '--', color='k', linewidth=1)

    state = env.reset0()
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(p.shape[0]):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = safe_ddpg_agent_list[i].policy_net.get_action(np.asarray([state[i]]))#+np.random.normal(0, 0.05)
            action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action - np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_load(action, p[step],q[step],pv_p[step])
    
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    
    for i in range(num_agent):    
        safes=axs[2].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], label = f'Bus {injection_bus[i]}', linewidth=1.5)
    axs[2].plot(range(len(action_list)), [0.95]*len(action_list), '--', color='k', linewidth=1)
    axs[2].plot(range(len(action_list)), [1.05]*len(action_list), '--', color='k', linewidth=1)
    # axs[2].get_yaxis().set_visible(False)
    # leg1 = plt.legend(safe_a_plt, safe_name, loc='lower left')
    axs[0].legend(loc='upper left', prop={"size":10})
    # box = axs[1].get_position()
    # axs[1].set_position([box.x0, box.y0,
    #              box.width, box.height * 0.8])
    # box = axs[2].get_position()
    # axs[2].set_position([box.x0, box.y0,
    #              box.width, box.height * 0.8])
    # axs[1].legend(loc='upper center', bbox_to_anchor=(1.1, 1.4),
    #       fancybox=True, shadow=True, ncol=6, prop={"size":10})
    # axs[2].legend(loc='lower right', prop={"size":10},ncol=2)
    axs[0].set_xlabel('Time (Hour)')   
    axs[1].set_xlabel('Time (Hour)')  
    axs[2].set_xlabel('Time (Hour)')  
    # axs[2].get_yaxis().set_visible(False)
    axs[1].set_yticks([0.95,1.00,1.05,1.10])
    axs[1].set_yticklabels(['0.95','1.00','1.05','1.10'])
    axs[2].set_yticks([0.95,1.00,1.05,1.10])
    axs[2].set_yticklabels(['0.95','1.00','1.05','1.10'])
    axs[0].set_xticks(np.arange(0,len(action_list),3600))
    axs[0].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
    axs[1].set_xticks(np.arange(0,len(action_list),3600))
    axs[1].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
    axs[2].set_xticks(np.arange(0,len(action_list),3600))
    axs[2].set_xticklabels(['00:00','06:00','12:00','18:00','24:00'], fontsize=13)
    axs[0].set_ylabel('Power (MW/MVar)', fontsize=15)   
    axs[1].set_ylabel('Bus voltage (p.u.)', fontsize=15)  
    axs[2].set_ylabel('Bus voltage (p.u.)', fontsize=15)  
    plt.show()


plot_traj_no_leg(p['p'],q['q'],np.resize(PV_p['actual_PV_profile'],(q['q'].shape[0],1)))