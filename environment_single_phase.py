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

class VoltageCtrl_nonlinear(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95):
        self.network =  pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
    
    def step(self, action): 
        
        done = False 
        
        reward = float(-50*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i+1, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done

    
    def step_Preward(self, action, p_action): 
        
        done = False 
        
        reward = float(-10*LA.norm(p_action) -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf)))
        
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        
        for i in range(agent_num):
            reward_sep[i] = float(-50*LA.norm(p_action[i])**2 -100*LA.norm(np.clip(self.state[i]-self.vmax, 0, np.inf))**2
                           - 100*LA.norm(np.clip(self.vmin-self.state[i], 0, np.inf))**2)              
        
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i+1, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        
        return self.state, reward, reward_sep, done

    
    def step_load(self, action, load_p, load_q): #state-transition with specific load
        
        done = False 
        
        reward = float(-50*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)
        
        #adjust power consumption at the load bus
        for i in range(self.env.agentnum):
            self.network.load.at[i, 'p_mw'] = load_p[i]
            self.network.load.at[i, 'q_mvar'] = load_q[i]
           
        #adjust reactive power inj at the PV bus
        for i in range(self.env.agentnum):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        state_all = self.network.res_bus.vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, state_all, reward, done
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0, 1])
        senario = 1
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.sgen.at[2, 'p_mw'] = -0.6*np.random.uniform(10, 30)
            self.network.sgen.at[3, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(2, 8)
            self.network.sgen.at[5, 'p_mw'] = -0.4*np.random.uniform(2, 8)

        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = 0.5*np.random.uniform(2, 10)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(5, 40)
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 14)
            self.network.sgen.at[4, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
            self.network.sgen.at[5, 'p_mw'] = 0.4*np.random.uniform(2, 14) 
        
        else: #mixture (this is used only during testing)
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[1, 'p_mw'] = -2*np.random.uniform(2, 3)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(15, 35)
            self.network.sgen.at[2, 'q_mvar'] = 0.1*self.network.sgen.at[2, 'p_mw']
            self.network.sgen.at[3, 'p_mw'] = 0.2*np.random.uniform(2, 12)
            self.network.sgen.at[4, 'p_mw'] = -2*np.random.uniform(2, 8) 
            self.network.sgen.at[5, 'p_mw'] = 0.2*np.random.uniform(2, 12) 
            self.network.sgen.at[5, 'q_mvar'] = 0.2*self.network.sgen.at[5, 'p_mw']
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state
    
    def reset0(self, seed=1): #reset voltage to nominal value
        
        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

def create_56bus():
    pp_net = pp.converter.from_mpc('SCE_56bus.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 17, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 20, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 29, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 44, p_mw = 2, q_mvar=0)
    pp.create_sgen(pp_net, 52, p_mw = 2, q_mvar=0)    
    return pp_net

if __name__ == "__main__":
    net = create_56bus()
    # print(net.sgen)
    injection_bus = np.array([18, 21, 30, 45, 53])-1  
    env = VoltageCtrl_nonlinear(net, injection_bus)
    # print(env.network.load)
    # exit(0)
    state_list = []
    for i in range(200):
        state = env.reset(i)
        state_list.append(state)
    state_list = np.array(state_list)
    fig, axs = plt.subplots(1, 5, figsize=(15,3))
    for i in range(5):
        axs[i].hist(state_list[:,i])
    plt.show()