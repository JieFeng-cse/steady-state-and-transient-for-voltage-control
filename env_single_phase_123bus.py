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

def bowl(vs, v_ref=1.0, scale=.1):
    def normal(v, loc, scale):
        return 1 / np.sqrt(2 * np.pi * scale**2) * np.exp( - 0.5 * np.square(v - loc) / scale**2 )
    def _bowl(v):
        if np.abs(v-v_ref) > 0.05:
            return 2 * np.abs(v-v_ref) - 0.095
        else:
            return - 0.01 * normal(v, v_ref, scale) + 0.04
    return np.array([_bowl(v) for v in vs])

class IEEE123bus(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95):
        self.network =  pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax #- 0.01
        self.vmin = vmin #+ 0.01
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
        self.init_state = np.ones(self.agentnum, )
        # self.C = np.asarray([0.1,0.2,0.3,0.3,0.5,0.7,1.0,0.7,1.0,1.0,1.0,1.0,0.5,0.7])*0.025
        self.C = np.asarray([0.2,0.25,0.1,0.3,0.3,0.2,0.2,0.3,0.9,0.7,0.7,0.7,0.6,0.7])*0.02
    
    def step(self, action): 
        
        done = False 
        
        reward = float(-50*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
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
        
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        
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
        senario = np.random.choice([0,1])
        senario=1
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = -0.8*np.random.uniform(15, 60)
            # self.network.sgen.at[0, 'q_mvar'] = -0.8*np.random.uniform(10, 300)
            self.network.sgen.at[1, 'p_mw'] = -0.8*np.random.uniform(10, 50)
            self.network.sgen.at[2, 'p_mw'] = -0.8*np.random.uniform(10, 55)
            self.network.sgen.at[3, 'p_mw'] = -0.8*np.random.uniform(10, 30)
            self.network.sgen.at[4, 'p_mw'] = -0.6*np.random.uniform(1, 35)
            self.network.sgen.at[5, 'p_mw'] = -0.5*np.random.uniform(2, 25)
            self.network.sgen.at[6, 'p_mw'] = -0.8*np.random.uniform(2, 30)
            self.network.sgen.at[7, 'p_mw'] = -0.9*np.random.uniform(1, 10)
            self.network.sgen.at[8, 'p_mw'] = -0.7*np.random.uniform(1, 15)
            self.network.sgen.at[9, 'p_mw'] = -0.5*np.random.uniform(1, 30)
            self.network.sgen.at[10, 'p_mw'] = -0.3*np.random.uniform(1, 20)
            self.network.sgen.at[11, 'p_mw'] = -0.5*np.random.uniform(1, 20)
            self.network.sgen.at[12, 'p_mw'] = -0.4*np.random.uniform(1, 20)
            self.network.sgen.at[13, 'p_mw'] = -0.4*np.random.uniform(2, 10)
            #not real controllers
            self.network.sgen.at[14, 'p_mw'] = -0.4*np.random.uniform(10, 20)
            self.network.sgen.at[15, 'p_mw'] = -0.8*np.random.uniform(10, 20)
            self.network.sgen.at[16, 'p_mw'] = -0.8*np.random.uniform(10, 20)


        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            
            self.network.sgen.at[0, 'p_mw'] = 0.8*np.random.uniform(15, 60)
            # self.network.sgen.at[0, 'q_mvar'] = 0.6*np.random.uniform(5, 300)
            self.network.sgen.at[1, 'p_mw'] = 0.8*np.random.uniform(15, 50)
            self.network.sgen.at[2, 'p_mw'] = 0.8*np.random.uniform(20, 60)
            self.network.sgen.at[3, 'p_mw'] = 0.8*np.random.uniform(10, 34)
            self.network.sgen.at[4, 'p_mw'] = 0.8*np.random.uniform(2, 20)
            self.network.sgen.at[5, 'p_mw'] = 0.8*np.random.uniform(2, 80)
            self.network.sgen.at[6, 'p_mw'] = 0.8*np.random.uniform(10, 80)
            self.network.sgen.at[7, 'p_mw'] = 0.8*np.random.uniform(5, 50)
            self.network.sgen.at[8, 'p_mw'] = 0.7*np.random.uniform(2, 30)
            self.network.sgen.at[9, 'p_mw'] = 0.5*np.random.uniform(2, 30)
            self.network.sgen.at[10, 'p_mw'] = 0.4*np.random.uniform(1, 40)
            self.network.sgen.at[11, 'p_mw'] = 0.5*np.random.uniform(1, 30)
            self.network.sgen.at[12, 'p_mw'] = 0.5*np.random.uniform(1, 30)
            self.network.sgen.at[13, 'p_mw'] = 0.5*np.random.uniform(1, 24)
            #not real controllers
            self.network.sgen.at[14, 'p_mw'] = 0.5*np.random.uniform(15, 25)
            self.network.sgen.at[15, 'p_mw'] = 0.8*np.random.uniform(10, 50)
            self.network.sgen.at[16, 'p_mw'] = 0.8*np.random.uniform(10, 20)
        
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

def create_123bus():
    pp_net = pp.converter.from_mpc('pandapower models/pandapower models/case_123.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 9, p_mw = 1.5, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 15, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 19, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 32, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 35, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 47, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 58, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 65, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 74, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 82, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 91, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 103, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 60, p_mw = 1, q_mvar=0) #node 114 in the png
    
    #only for reset
    pp.create_sgen(pp_net, 13, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 14, p_mw = 1, q_mvar=0)
    pp.create_sgen(pp_net, 18, p_mw = 1, q_mvar=0)

    return pp_net

if __name__ == "__main__":
    net = create_123bus()
    # print(net.shunt)
    # exit(0)
    injection_bus = np.array([10, 11, 16, 20, 33, 36, 48, 59, 66, 75, 83, 92, 104, 61])-1 #11, 36, 75,/ 1,5,9
    env = IEEE123bus(net, injection_bus)
    state_list = []
    for i in range(200):
        state = env.reset(i)
        state_list.append(state)
    state_list = np.array(state_list)
    fig, axs = plt.subplots(2, 7, figsize=(15,3))
    for i in range(2):
        for j in range(7):
            axs[i,j].hist(state_list[:,i*7+j])
    plt.show()

