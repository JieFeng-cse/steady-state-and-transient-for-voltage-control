import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import linalg as LA

from scipy.io import loadmat

import torch
import torch.nn.functional as F
import argparse

from environment_single_phase import create_56bus, VoltageCtrl_nonlinear
from env_single_phase_13bus import IEEE13bus, create_13bus
from env_single_phase_123bus import IEEE123bus, create_123bus
from safeDDPG import ValueNetwork, SafePolicyNetwork, DDPG, PolicyNetwork, SafePolicy3phase, LinearPolicy
from IEEE_13_3p import IEEE13bus3p, create_13bus3p
from tqdm import tqdm


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Single Phase Safe DDPG')
parser.add_argument('--env_name', default="13bus",
                    help='name of the environment to run')
parser.add_argument('--algorithm', default='safe-ddpg', help='name of algorithm')
parser.add_argument('--safe_type', default='three_single')
parser.add_argument('--safe_method', default='safe-flow') 
parser.add_argument('--use_safe_flow', default='True') 
parser.add_argument('--use_gradient', default='True') 
args = parser.parse_args()
seed = 10
torch.manual_seed(seed)
plt.rcParams['font.size'] = '20'

"""
Create Agent list and replay buffer
"""
max_ac = 0.3
ph_num = 1
slope = 2
type_name = 'single-phase'

if args.env_name == '123bus':
    pp_net = create_123bus()
    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61])-1
    env = IEEE123bus(pp_net, injection_bus)
    num_agent = 14
    # Q_limit = np.asarray([[-15,15],[-10,10],[-13,13],[-7,7],[-6,6],[-3.5,3.5],[-7,7],[-2.5,2.5],[-3,3],[-4.5,4.5],[-1.5,1.5],[-3,3],[-2.4,2.4],[-1.2,1.2]])
    Q_limit = np.asarray([[-21.6,21.6],[-18,18],[-21.6,21.6],[-10.8,10.8],[-9.45,9.45],[-20,20],[-20,20],[-16,16],[-4.725,4.725],[-7.2,7.2],[-7.2,7.2],[-6.75,6.75],[-6.75,6.75],[-5.4,5.4]])
    # C = np.asarray([0.1,0.2,0.3,0.3,0.5,0.7,1.0,0.7,1.0,1.0,1.0,1.0,0.5,0.7])*0.025
    C = np.asarray([0.2,0.25,0.1,0.3,0.3,0.2,0.2,0.3,0.9,0.7,0.7,0.7,0.6,0.7])*0.02
    if args.safe_method == 'project':
        alpha = 1
    else:
        alpha =0.5

obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 100



alpha2_list = []
alpha5_list = []
alpha8_list = []

#alpha = 0.2
for i in range(num_agent):    
    safe_ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    
    safe_ddpg_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
        up=Q_limit[i,1],low=Q_limit[i,0],alpha=0.2,node_cost=C[i],\
            use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
    safe_ddpg_agent2 = DDPG(policy_net=safe_ddpg_policy_net, value_net=safe_ddpg_value_net,
                 target_policy_net=safe_ddpg_policy_net, target_value_net=safe_ddpg_value_net)  
    alpha2_list.append(safe_ddpg_agent2) 

#alpha = 0.5
for i in range(num_agent):    
    safe_ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    
    safe_ddpg_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
        up=Q_limit[i,1],low=Q_limit[i,0],alpha=0.5,node_cost=C[i],\
            use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
    safe_ddpg_agent5 = DDPG(policy_net=safe_ddpg_policy_net, value_net=safe_ddpg_value_net,
                 target_policy_net=safe_ddpg_policy_net, target_value_net=safe_ddpg_value_net) 
    alpha5_list.append(safe_ddpg_agent5)   

#alpha = 0.8
for i in range(num_agent):    
    safe_ddpg_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)    
    safe_ddpg_policy_net = SafePolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, \
        up=Q_limit[i,1],low=Q_limit[i,0],alpha=0.8,node_cost=C[i],\
            use_gradient=(args.use_gradient=='True'), safe_flow=(args.use_safe_flow=='True')).to(device)
    safe_ddpg_agent8 = DDPG(policy_net=safe_ddpg_policy_net, value_net=safe_ddpg_value_net,
                 target_policy_net=safe_ddpg_policy_net, target_value_net=safe_ddpg_value_net) 
    alpha8_list.append(safe_ddpg_agent8)   

#load the models
for i in range(num_agent):
    safe_ddpg_policynet_dict2 = torch.load(f'checkpoints/{type_name}/{args.env_name}/safe-ddpg/0.2safe_policy_net_checkpoint_a{i}.pth')
    alpha2_list[i].policy_net.load_state_dict(safe_ddpg_policynet_dict2)

    safe_ddpg_policynet_dict5 = torch.load(f'checkpoints/{type_name}/{args.env_name}/safe-ddpg/safe-flow_policy_net_checkpoint_a{i}.pth')
    alpha5_list[i].policy_net.load_state_dict(safe_ddpg_policynet_dict5)

    safe_ddpg_policynet_dict8 = torch.load(f'checkpoints/{type_name}/{args.env_name}/safe-ddpg/0.8safe_policy_net_checkpoint_a{i}.pth')
    alpha8_list[i].policy_net.load_state_dict(safe_ddpg_policynet_dict8)

def plot_traj_123(seed):
    color_set = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, axs = plt.subplots(1, 2, figsize=(13,5))
    ddpg_plt=[]
    ddpg_a_plt=[]
    state = env.reset(seed)
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(100):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = alpha2_list[i].policy_net.get_action(np.asarray([state[i]]),last_action[i])
            action.append(action_agent)

        # PI policy    
        action = last_action + np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (action-last_action))
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    for idx,i in enumerate([2,8]): 
        axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '--', label = r'$\alpha=0.2$'+f' at bus {injection_bus[i]+1}', linewidth=2,color=color_set[idx])
        axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], '--', label = r'$\alpha=0.2$'+f' at bus {injection_bus[i]+1}', linewidth=2,color=color_set[idx])

    state = env.reset(seed)
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(100):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = alpha5_list[i].policy_net.get_action(np.asarray([state[i]]),last_action[i])
            action.append(action_agent)

        # PI policy    
        action = last_action + np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (action-last_action))
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    
    # lb = axs[0].plot(range(len(action_list)), [0.95]*len(action_list), linestyle='--', dashes=(5, 10), color='g', label='lower bound')
    # ub = axs[0].plot(range(len(action_list)), [1.05]*len(action_list), linestyle='--', dashes=(5, 10), color='r', label='upper bound')
    for idx,i in enumerate([2,8]):    #[2,5,8]
        dps = axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-.', label = r'$\alpha=0.5$'+f' at bus {injection_bus[i]+1}', linewidth=2,color=color_set[idx])
        dpa = axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], '-.', label = r'$\alpha=0.5$'+f' at bus {injection_bus[i]+1}', linewidth=2,color=color_set[idx])
        ddpg_plt.append(dps)
        ddpg_a_plt.append(dpa)

    state = env.reset(seed)
    episode_reward = 0
    last_action = np.zeros((num_agent,1))
    action_list=[]
    state_list =[]
    state_list.append(state)
    for step in range(100):
        action = []
        for i in range(num_agent):
            # sample action according to the current policy and exploration noise
            action_agent = alpha8_list[i].policy_net.get_action(np.asarray([state[i]]),last_action[i])#+np.random.normal(0, 0.05)
            # action_agent = (np.maximum(state[i]-1.05, 0)-np.maximum(0.95-state[i], 0)).reshape((1,))*2
            # action_agent = np.clip(action_agent, -max_ac, max_ac)
            action.append(action_agent)

        # PI policy    
        action = last_action + np.asarray(action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step_Preward(action, (action-last_action))
        action_list.append(action)
        state_list.append(next_state)
        last_action = np.copy(action)
        state = next_state
    for idx,i in enumerate([2,8]): 
        axs[0].plot(range(len(action_list)), np.array(state_list)[:len(action_list),i], '-', label = r'$\alpha=0.8$'+f' at bus {injection_bus[i]+1}', linewidth=2,color=color_set[idx])
        axs[1].plot(range(len(action_list)), np.array(action_list)[:,i], label = r'$\alpha=0.8$'+f' at bus {injection_bus[i]+1}', linewidth=2,color=color_set[idx])

    
    matplotlib.rcParams['text.usetex']=True
    axs[0].plot(range(len(action_list)),np.ones_like(np.array(state_list)[:len(action_list),0])*1.05, '--',linewidth=2,color='dimgray')  #,label=r'$\bar{v}$'
    # axs[1].plot(range(len(action_list)),np.ones_like(np.array(state_list)[:len(action_list),0])*(-21.6), ':',linewidth=2,label=r'$\underline{q}$'+f' at bus {injection_bus[2]+1}',color=color_set[4]) 
    axs[1].plot(range(len(action_list)),np.ones_like(np.array(state_list)[:len(action_list),0])*(-4.75), '--',linewidth=2,color='dimgray') #,label=r'$\underline{q}$'+f' at bus {injection_bus[8]+1}'
    # leg1 = plt.legend(safe_a_plt, safe_name, loc='lower left')
    # axs[0].legend(loc='lower left', prop={"size":20})
    # axs[1].legend(loc='lower left', prop={"size":20})
    # box = axs[0].get_position()
    # axs[0].set_position([box.x0-0.05*box.width, box.y0+0.09*box.height,
    #                 box.width* 0.9, box.height*0.9])
    # box = axs[1].get_position()
    # axs[1].set_position([box.x0+0.05*box.width, box.y0+0.09*box.height,
    #                 box.width* 0.9, box.height*0.9])
    box = axs[0].get_position()
    axs[0].set_position([box.x0-0.05*box.width, box.y0+0.4*box.height,
                    box.width* 0.95, box.height*0.7])
    box = axs[1].get_position()
    axs[1].set_position([box.x0+0.05*box.width, box.y0+0.4*box.height,
                    box.width* 0.95, box.height*0.7])
    axs[0].legend(loc='lower center', bbox_to_anchor=(1.2, -0.7),
        fancybox=True, ncol=3, prop={"size":15})
    # axs[0].legend(loc='right', bbox_to_anchor=(3.05, 0.4),
    #     fancybox=True, shadow=True, ncol=1,prop={"size":13})
    # axs[0].legend(loc='upper right', prop={"size":13})
    # axs[1].legend(loc='upper right', prop={"size":13})
    axs[0].set_xlabel('Iteration Steps')   
    axs[1].set_xlabel('Iteration Steps')  
    axs[0].set_ylabel('Bus Voltage [p.u.]')   
    axs[1].set_ylabel('q Injection [MVar]')  
    # axs[0].set_title(f'{seed}')
    plt.savefig('alpha.png')

if __name__ == "__main__":
    
    plot_traj_123(5)