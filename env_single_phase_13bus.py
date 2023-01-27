import numpy as np
from numpy import linalg as LA
import gym
import os
import random
import sys
from gym import spaces
from gym.utils import seeding
import copy
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math

class IEEE13bus(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95, all_bus=False):
        self.network =  pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        if self.agentnum == 12:
            all_bus=True
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        self.all_bus = all_bus
        
        self.state = np.ones(self.agentnum, )
        self.init_state = np.ones(self.agentnum, )
        self.C = np.asarray([0.7,0.5,0.6])*0.15
    
    def step(self, action): 
        
        done = False 
        
        reward = float(-50*LA.norm(action)**2 - 100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done

    
    def step_Preward(self, action, p_action): 
        
        done = False 
        #safe-ddpg reward
        # reward = float(-10*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
        #                - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        #why in this part originally it is not square?
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )

        # for i in range(agent_num):
        #     if (self.state[i]>1.0 and self.state[i]<1.05):
        #         reward_sep[i] = float(-0*LA.norm(p_action[i],1) -0*LA.norm([np.clip(self.state[i]-self.vmax, -np.inf, 0)],2)**2)   
        #     elif (self.state[i]>0.95 and self.state[i]<1.0):
        #         reward_sep[i] = float(-0*LA.norm(p_action[i],1) -0*LA.norm([np.clip(self.vmin-self.state[i], -np.inf, 0)],2)**2)   
        #     elif self.state[i]<0.95:
        #         reward_sep[i] = float(-self.C[i]*LA.norm(p_action[i],1) -100*LA.norm([np.clip(self.vmin-self.state[i], 0, np.inf)],2)**2) 
        #     elif self.state[i]>1.05:
        #         reward_sep[i] = float(-self.C[i]*LA.norm(p_action[i],1) -100*LA.norm([np.clip(self.state[i]-self.vmax, 0, np.inf)],2)**2) 
        # reward = np.sum(reward_sep)
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        reward = -np.squeeze(0.5*action.T@np.diag(self.C)@action + 0.5*action.T@(np.square(self.state)+np.square(self.init_state)\
            -2*np.ones_like(self.state)))
        for i in range(agent_num):
            reward_sep[i] = -(0.5*np.square(action[i])*self.C[i]+0.5*action[i]*(np.square(self.state[i])+np.square(self.init_state[i])-2))
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        return self.state, reward, reward_sep, done

    
    def step_load(self, action, load_p, load_q, pv_p): #state-transition with specific load
        
        done = False 
        
        reward = float(-50*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        
        #adjust power consumption at the load bus
        load_idx = [1,5,10]
        # for i in range(len(self.network.load)):
        self.network.load.at[0, 'p_mw'] = load_p*0.01
        self.network.load.at[0, 'q_mvar'] = load_q*0.02
        self.network.load.at[1, 'p_mw'] = load_p*0.03
        self.network.load.at[1, 'q_mvar'] = load_q*0.01
        self.network.load.at[2, 'p_mw'] = load_p*0.03
        self.network.load.at[2, 'q_mvar'] = load_q*0.02
        self.network.load.at[3, 'p_mw'] = load_p*0.01
        self.network.load.at[3, 'q_mvar'] = load_q*0.01
        self.network.load.at[4, 'p_mw'] = load_p*0.03
        self.network.load.at[4, 'q_mvar'] = load_q*0.02
        self.network.load.at[5, 'p_mw'] = load_p*0.02
        self.network.load.at[5, 'q_mvar'] = load_q*0.01
        self.network.load.at[6, 'p_mw'] = load_p*0.01
        self.network.load.at[6, 'q_mvar'] = load_q*0.01
        self.network.load.at[7, 'p_mw'] = load_p*0.03
        self.network.load.at[7, 'q_mvar'] = load_q*0.02
        self.network.load.at[8, 'p_mw'] = load_p*0.01
        self.network.load.at[8, 'q_mvar'] = load_q*0.01
        # for i in range(len(self.injection_bus)):
        #     self.network.load.at[i, 'p_mw'] = load_p*0.1
        #     self.network.load.at[i, 'q_mvar'] = load_q*0.1
        
           
        #adjust reactive power inj at the PV bus
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i]

        #adjust reactive power inj at the PV bus
        # for i in range(len(self.injection_bus)):
            # self.network.sgen.at[i, 'p_mw'] = pv_p*0.15
            
        # self.network.sgen.at[0, 'p_mw'] = pv_p*0.4
        # self.network.sgen.at[1, 'p_mw'] = pv_p*0.4
        # self.network.sgen.at[2, 'p_mw'] = pv_p*0.4
        # self.network.sgen.at[4, 'p_mw'] = 0.2*pv_p
        # self.network.sgen.at[5, 'q_mvar'] = 0.3*pv_p
        self.network.sgen.at[0, 'p_mw'] = pv_p*0.15
        self.network.sgen.at[1, 'p_mw'] = pv_p*0.2
        self.network.sgen.at[2, 'p_mw'] = pv_p*0.2
        self.network.sgen.at[3, 'p_mw'] = pv_p*0.1
        self.network.sgen.at[4, 'p_mw'] = 0.3*pv_p
        self.network.sgen.at[5, 'p_mw'] = 0.1*pv_p
        self.network.sgen.at[6, 'p_mw'] = pv_p*0.01
        self.network.sgen.at[7, 'p_mw'] = pv_p*0.05
        self.network.sgen.at[8, 'p_mw'] = pv_p*0.01
        self.network.sgen.at[9, 'p_mw'] = pv_p*0.01
        self.network.sgen.at[10, 'p_mw'] = pv_p*0.1
        self.network.sgen.at[11, 'p_mw'] = pv_p*0.01

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, state_all, reward, done
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        # senario = 0
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -0.5*np.random.uniform(1, 7)
            self.network.sgen.at[1, 'p_mw'] = -0.8*np.random.uniform(1, 4)
            self.network.sgen.at[2, 'p_mw'] = -0.3*np.random.uniform(1, 5)
            if self.all_bus:
                for i in range(len(self.injection_bus)):
                    self.network.sgen.at[i, 'p_mw'] = -0.3*np.random.uniform(1, 2.5)
        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            # for i in range(len(self.injection_bus)):
            #     self.network.sgen.at[i, 'p_mw'] = 0.5*np.random.uniform(0, 2)
            self.network.sgen.at[0, 'p_mw'] = np.random.uniform(0.5, 4)
            self.network.sgen.at[1, 'p_mw'] = np.random.uniform(1, 4.51)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(0, 5)

            # self.network.sgen.at[3, 'q_mvar'] = 0.3*np.random.uniform(0, 0.2)
            self.network.sgen.at[3, 'p_mw'] = 0.6*np.random.uniform(0, 2)
            self.network.sgen.at[4, 'p_mw'] = 0.6*np.random.uniform(2, 3)
            # self.network.sgen.at[5, 'q_mvar'] = 0.4*np.random.uniform(0, 10)
            self.network.sgen.at[5, 'p_mw'] = 0.7*np.random.uniform(0, 2)
            
            self.network.sgen.at[10, 'p_mw'] = np.random.uniform(0.2, 3)
            self.network.sgen.at[11, 'p_mw'] = np.random.uniform(0, 1.5)
            #for all buses scheme
            if self.all_bus:
                self.network.sgen.at[6, 'p_mw'] = 0.5*np.random.uniform(1, 2)
                self.network.sgen.at[7, 'p_mw'] = 0.2*np.random.uniform(1, 3)
                self.network.sgen.at[8, 'p_mw'] = 0.2*np.random.uniform(2, 3)
                self.network.sgen.at[9, 'p_mw'] = np.random.uniform(0.1, 0.5)
        
        else: #mixture (this is used only during testing)
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -2*np.random.uniform(2, 3)
            self.network.sgen.at[1, 'p_mw'] = np.random.uniform(15, 35)
            self.network.sgen.at[1, 'q_mvar'] = 0.1*self.network.sgen.at[1, 'p_mw']
            self.network.sgen.at[2, 'p_mw'] = 0.2*np.random.uniform(2, 12)
            
        
        pp.runpp(self.network, algorithm='bfsw')
        # if not self.network['Converged_in_100_r']:
        #     print('fialed')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        self.init_state = np.copy(self.state)
        return self.state
    
    def reset0(self, seed=1): #reset voltage to nominal value
        
        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

def create_13bus():
    pp_net = pp.converter.from_mpc('pandapower models/pandapower models/case_13.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 2, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 9, p_mw = 0, q_mvar=0)

    pp.create_sgen(pp_net, 1, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 4, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 5, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 6, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 8, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 11, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 12, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 1, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 2, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 5, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 3, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 4, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 7, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 8, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 9, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 6, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 10, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 11, p_mw = 0, q_mvar=0)
    # pp.create_sgen(pp_net, 12, p_mw = 0, q_mvar=0)
    
    # # In the original IEEE 13 bus system, there is no load in bus 3, 7, 8. 
    # # Add the load to corresponding node for dimension alignment in RL training
    pp.create_load(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 8, p_mw = 0, q_mvar=0)

    return pp_net

if __name__ == "__main__":
    net = create_13bus()
    injection_bus = np.array([2, 7, 9])
    # injection_bus = np.array([1,2,3,4,5,6, 7, 8,9,10,11,12])
    env = IEEE13bus(net, injection_bus)
    # print(env.reset0())
    # for i in range(7):
    #     state,_,_,_=env.step_load([0,0,0],0,0,i)
    #     print(state)
    state_list = []
    for i in range(200):
        state = env.reset(i)
        state_list.append(state)
    state_list = np.array(state_list)
    fig, axs = plt.subplots(1, 3, figsize=(6,3))
    for i in range(3):
        axs[i].hist(state_list[:,i])
    plt.show()
    



