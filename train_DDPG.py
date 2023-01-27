### import collections
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
from env_three_phase_eu import Three_Phase_EU, create_eu_lv
from IEEE_13_3p import IEEE13bus3p, create_13bus3p
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, ReplayBuffer, ReplayBufferPI, PolicyNetwork, SafePolicy3phase, LinearPolicy
from ICNN import ICNN,convex_monotone_network

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Single Phase Safe DDPG')
parser.add_argument('--env_name', default="13bus",
                    help='name of the environment to run')
parser.add_argument('--algorithm', default='safe-ddpg', help='name of algorithm')
parser.add_argument('--status', default='train')
parser.add_argument('--safe_type', default='three_single') #loss, dd
parser.add_argument('--safe_method', default='safe-flow') 
parser.add_argument('--use_safe_flow', default='True') 
parser.add_argument('--use_gradient', default='True') 
args = parser.parse_args()
seed = 10
torch.manual_seed(seed)

"""
Create Agent list and replay buffer
"""
vlr = 2e-4
plr = 1e-4
ph_num = 1
max_ac = 0.3
alpha = 0.7
if args.env_name == '56bus':
    pp_net = create_56bus()
    injection_bus = np.array([18, 21, 30, 45, 53])-1  
    env = VoltageCtrl_nonlinear(pp_net, injection_bus)
    num_agent = 5

if args.env_name == '13bus':
    pp_net = create_13bus()
    injection_bus = np.array([2, 7, 9])
    # injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    env = IEEE13bus(pp_net, injection_bus)
    num_agent = len(injection_bus)
    Q_limit = np.asarray([[-1.0,1.0],[-1.0,0.8],[-1.0,0.6]])
    C = np.asarray([0.7,0.5,0.6])*0.15    

if args.env_name == '123bus':
    max_ac = 0.8
    pp_net = create_123bus()
    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61])-1
    env = IEEE123bus(pp_net, injection_bus)
    num_agent = 14
    Q_limit = np.asarray([[-15,15],[-10,10],[-13,13],[-7,7],[-6,6],[-3.5,3.5],[-7,7],[-2.5,2.5],[-3,3],[-4.5,4.5],[-1.5,1.5],[-3,3],[-2.4,2.4],[-1.2,1.2]])
    C = np.asarray([0.1,0.2,0.3,0.3,0.5,0.7,1.0,0.7,1.0,1.0,1.0,1.0,0.5,0.7])*0.025
    if args.algorithm == 'safe-ddpg':
        plr = 1.5e-4

if args.env_name == '13bus3p':
    # injection_bus = np.array([675,633,680])
    injection_bus = np.array([633,634,671,645,646,692,675,611,652,632,680,684])
    pp_net, injection_bus_dict = create_13bus3p(injection_bus) 
    max_ac = 0.5
    env = IEEE13bus3p(pp_net,injection_bus_dict)
    num_agent = len(injection_bus)
    ph_num=3
    if args.algorithm == 'safe-ddpg':
        plr = 1e-4
    if args.algorithm == 'ddpg':
        plr = 5e-5



obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100
if ph_num == 3:
    type_name = 'three-phase'
else:
    type_name = 'single-phase'

agent_list = []
replay_buffer_list = []

for i in range(num_agent):
    if ph_num == 3:
        obs_dim = len(env.injection_bus[env.injection_bus_str[i]])
        action_dim = obs_dim
    value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    if args.algorithm == 'safe-ddpg' and not ph_num == 3:
        policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
            up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[i],\
                use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
        target_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
            up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[i],\
                use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
    elif args.algorithm == 'safe-ddpg' and ph_num == 3 and args.safe_type == 'three_single':
        policy_net = SafePolicy3phase(env, obs_dim, action_dim, hidden_dim, env.injection_bus_str[i]).to(device)
        target_policy_net = SafePolicy3phase(env, obs_dim, action_dim, hidden_dim, env.injection_bus_str[i]).to(device)
    elif args.algorithm == 'linear':
        policy_net = LinearPolicy(env=env, ph_num=ph_num, \
            up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[i],\
                use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
        target_policy_net = LinearPolicy(env=env, ph_num=ph_num, \
            up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[i],\
                use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
    else:
        policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
            up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[i],\
                use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
        target_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
            up=Q_limit[i,1],low=Q_limit[i,0],alpha=alpha,node_cost=C[i],\
                use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)

    target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    

    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(param.data)

    agent = DDPG(policy_net=policy_net, value_net=value_net,
                 target_policy_net=target_policy_net, target_value_net=target_value_net, value_lr=vlr, policy_lr=plr)
    
    replay_buffer = ReplayBufferPI(capacity=1000000)
    
    agent_list.append(agent)
    replay_buffer_list.append(replay_buffer)

if args.status =='train':
    FLAG = 1
else:
    FLAG = 0
def get_id(phases):
    if phases == 'abc':
        id = [0,1,2]
    elif phases == 'ab':
        id = [0,1]
    elif phases == 'ac':
        id = [0,2]
    elif phases == 'bc':
        id = [1,2]
    elif phases == 'a':
        id = [0]
    elif phases == 'b':
        id = [1]
    elif phases == 'c':
        id = [2]
    else:
        print("error!")
        exit(0)
    return id

if (FLAG ==0): 
    # load trained policy
    for i in range(num_agent):
        if ph_num == 3 and args.algorithm=='safe-ddpg':
            valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/value_net_checkpoint_a{i}.pth')
            policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/policy_net_checkpoint_a{i}.pth')
        else:
            valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_method}_value_net_checkpoint_a{i}.pth')
            policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_method}_policy_net_checkpoint_a{i}.pth')
        agent_list[i].value_net.load_state_dict(valuenet_dict)
        agent_list[i].policy_net.load_state_dict(policynet_dict)

elif (FLAG ==1):
    # training episode
    # for i in range(num_agent):
    #     if ph_num == 3 and args.algorithm=='safe-ddpg':
    #         valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type} copy/value_net_checkpoint_a{i}.pth')
    #         policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type} copy/policy_net_checkpoint_a{i}.pth')
    #     else:
    #         valuenet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm} copy/value_net_checkpoint_a{i}.pth')
    #         policynet_dict = torch.load(f'checkpoints/{type_name}/{args.env_name}/{args.algorithm} copy/policy_net_checkpoint_a{i}.pth')
    #     agent_list[i].value_net.load_state_dict(valuenet_dict)
    #     agent_list[i].policy_net.load_state_dict(policynet_dict) 
    #     for target_param, param in zip(agent_list[i].target_value_net.parameters(), agent_list[i].value_net.parameters()):
    #         target_param.data.copy_(param.data)

    #     for target_param, param in zip(agent_list[i].target_policy_net.parameters(), agent_list[i].policy_net.parameters()):
    #         target_param.data.copy_(param.data)

    if args.algorithm == 'safe-ddpg':
        num_episodes = 700    #13-3p
        # num_episodes = 700
    elif args.algorithm == 'ddpg':
        # num_episodes = 700 #123 2000 13-3p 700
        num_episodes = 2000 #13 300
    elif args.algorithm == 'linear':
        num_episodes = 700


    # trajetory length each episode
    num_steps = 30  
    if args.env_name =='123bus':
        num_steps = 60
    if args.env_name =='13bus3p':
        num_steps = 30
    batch_size = 256

    # if/not plot trained policy every # episodes
    rewards = []
    avg_reward_list = []
    for episode in range(num_episodes):
        state = env.reset(seed = episode)
        episode_reward = 0
        last_action = np.zeros((num_agent,ph_num)) #if single phase, 1, else ,3

        for step in range(num_steps):
            action = []
            action_p = []
            for i in range(num_agent):
                # sample action according to the current policy and exploration noise
                if ph_num==3:
                    action_agent = np.zeros(3)
                    phases = env.injection_bus[env.injection_bus_str[i]]
                    id = get_id(phases)
                    action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[i,id]])) \
                        + np.random.normal(0, max_ac)/np.sqrt(episode+1)
                    action_tmp = action_tmp.reshape(len(id),)  
                    for j in range(len(phases)):
                        action_agent[id[j]]=action_tmp[j]           
                    # action_p.append(action_agent)
                    action_agent = np.clip(action_agent, -max_ac, max_ac) 
                    action_p.append(action_agent)
                else:
                    action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]]), last_action[i]) \
                        + np.random.normal(0, max_ac)/np.sqrt(episode+1)
                    action_p.append(action_agent)                    
                action.append(action_agent)

            # PI policy    
            if ph_num == 3:
                action = last_action - np.asarray(action).reshape(-1,3) #.reshape(-1,3) #if eu, reshape
            else:
                action = last_action + np.asarray(action)

            # execute action a_t and observe reward r_t and observe next state s_{t+1}
            next_state, reward, reward_sep, done = env.step_Preward(action, action_p)
            
            if(np.min(next_state)<0.75): #if voltage violation > 25%, episode ends.
                break
            else:
                for i in range(num_agent): 
                    if ph_num == 1:
                        state_buffer = state[i].reshape(ph_num,) 
                        action_buffer = action[i].reshape(ph_num,)
                        last_action_buffer = last_action[i].reshape(ph_num,)
                        next_state_buffer = next_state[i].reshape(ph_num, )
                    else:
                        phases = env.injection_bus[env.injection_bus_str[i]]
                        id = get_id(phases)
                        state_buffer = state[i,id].reshape(len(phases),) 
                        action_buffer = action[i,id].reshape(len(phases),) 
                        last_action_buffer = last_action[i,id].reshape(len(phases),) 
                        next_state_buffer = next_state[i,id].reshape(len(phases),) 
                    # store transition (s_t, a_t, r_t, s_{t+1}) in R
                    replay_buffer_list[i].push(state_buffer, action_buffer, last_action_buffer,
                                               reward_sep[i], next_state_buffer, done) #_sep[i]

                    # update both critic and actor network
                    if ph_num == 3 and args.algorithm == 'safe-ddpg' and args.safe_type == 'loss':
                        if len(replay_buffer_list[i]) > batch_size:
                            agent_list[i].train_step_3ph(replay_buffer=replay_buffer_list[i], 
                                                    batch_size=batch_size)
                    else:
                        if len(replay_buffer_list[i]) > batch_size:
                            agent_list[i].train_step(replay_buffer=replay_buffer_list[i], 
                                                    batch_size=batch_size)

                if(done):
                    episode_reward += reward  
                    break #no break if 13bus3p
                else:
                    state = np.copy(next_state)
                    episode_reward += reward    

            last_action = np.copy(action)

        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-40:])
        if(episode%50==0):
            print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
        avg_reward_list.append(avg_reward)
    for i in range(num_agent):
        if ph_num == 3 and args.algorithm=='safe-ddpg':
            pth_value = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/value_net_checkpoint_a{i}.pth'
            pth_policy = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_type}/policy_net_checkpoint_a{i}.pth'
        else:
            pth_value = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_method}_value_net_checkpoint_a{i}.pth'
            pth_policy = f'checkpoints/{type_name}/{args.env_name}/{args.algorithm}/{args.safe_method}_policy_net_checkpoint_a{i}.pth'
        torch.save(agent_list[i].value_net.state_dict(), pth_value)
        torch.save(agent_list[i].policy_net.state_dict(), pth_policy)

else:
    raise ValueError("Model loading optition does not exist!")
# torch.save(torch.tensor(rewards),'rewards_ddpg_q1.pt')
# print(torch.tensor(rewards).shape)

fig, axs = plt.subplots(1, num_agent, figsize=(15,3))
for i in range(num_agent):
    # plot policy
    N = 50
    s_array = np.zeros(N,)
    
    a_array_baseline = np.zeros(N,)
    a_array = np.zeros((N,ph_num))
    
    for j in range(N):
        if ph_num ==1:
            state = np.array([0.8+0.01*j])
            s_array[j] = state
            action_baseline = (np.maximum(state-1.05, 0)-np.maximum(0.95-state, 0)).reshape((1,))
        else:
            state = np.resize(np.array([0.8+0.01*j]),(3))
            s_array[j] = state[0]        
            action_baseline = (np.maximum(state[0]-1.03, 0)-np.maximum(0.97-state[0], 0)).reshape((1,))
        if ph_num == 3: 
            action = np.zeros(3)
            phases = env.injection_bus[env.injection_bus_str[i]]
            id = get_id(phases)
            action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[id]])) 
            action_tmp = action_tmp.reshape(len(id),)  
            for p in range(len(phases)):
                action[id[p]]=action_tmp[p]
        else:
            action = agent_list[i].policy_net.get_action(np.asarray([state]),np.zeros_like([state])) 
        # action = np.clip(action, -max_ac, max_ac) 
        a_array_baseline[j] = -action_baseline[0]
        a_array[j] = action
    axs[i].plot(s_array, 2*a_array_baseline, '-.', label = 'Linear')
    for k in range(ph_num):        
        axs[i].plot(s_array, a_array[:,k], label = args.algorithm)
        axs[i].legend(loc='lower left')
plt.show()

## test policy
state = env.reset()
episode_reward = 0
last_action = np.zeros((num_agent,1))
action_list=[]
state_list =[]
reward_list = []
state_list.append(state)
for step in range(100):
    action = []
    for i in range(num_agent):
        # sample action according to the current policy and exploration noise
        if ph_num==3:
            action_agent = np.zeros(3)
            phases = env.injection_bus[env.injection_bus_str[i]]
            id = get_id(phases)
            action_tmp = agent_list[i].policy_net.get_action(np.asarray([state[i,id]])) 
            action_tmp = action_tmp.reshape(len(id),)  
            for i in range(len(phases)):
                action_agent[id[i]]=action_tmp[i]
            action_agent = np.clip(action_agent, -max_ac, max_ac) 
            action.append(action_agent)
        else:
            action_agent = agent_list[i].policy_net.get_action(np.asarray([state[i]]),last_action[i])
            action.append(action_agent)

    # PI policy    
    action = last_action + np.asarray(action)

    # execute action a_t and observe reward r_t and observe next state s_{t+1}
    next_state, reward, reward_sep, done = env.step_Preward(action, (action-last_action))
    reward_list.append(reward)
    if done:
        print("finished")
    action_list.append(action-last_action)
    state_list.append(next_state)
    last_action = np.copy(action)
    state = next_state
fig, axs = plt.subplots(1, num_agent+1, figsize=(15,3))
for i in range(num_agent):
    axs[i].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-.', label = 'states')
    # axs[i].plot(range(len(action_list)), np.array(action_list)[:,i], label = 'action')
    axs[i].legend(loc='lower left')
fig1, axs1 = plt.subplots(1, num_agent+1, figsize=(15,3))
for i in range(num_agent):
    axs1[i].plot(range(len(action_list)), np.array(action_list)[:len(action_list),i], '-.', label = 'actions')
    # axs[i].plot(range(len(action_list)), np.array(action_list)[:,i], label = 'action')
    axs1[i].legend(loc='lower left')
axs[num_agent].plot(range(len(reward_list)),reward_list)
plt.show()

# test success rate:
