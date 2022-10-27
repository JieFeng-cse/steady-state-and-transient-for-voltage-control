from unittest import skip
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
import pandapower.plotting as plot

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
import pandas as pd 
import math
import seaborn

class Three_Phase_EU(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95):
        self.network =  pp_net
        self.obs_dim = 3
        self.action_dim = 3
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin        
        self.state = np.ones((self.agentnum, 3))
    
    def step_Preward(self, action, p_action): 
        
        done = False 
        #safe-ddpg reward
        reward = float(-0*LA.norm(p_action)**2-1000*LA.norm(np.clip(self.state-self.vmax, 0, np.inf),1)
                       -1000*LA.norm(np.clip(self.vmin-self.state, 0, np.inf),1))
        #why in this part originally it is not square?
        # local reward
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )
        #just for ddpg
        for i in range(agent_num):
            reward_sep[i] = float(-50*LA.norm(p_action[i])**2 -100*LA.norm(np.clip(self.state[i]-self.vmax, 0, np.inf))**2 
                           - 100*LA.norm(np.clip(self.vmin-self.state[i], 0, np.inf))**2 )              
        
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.asymmetric_sgen.at[i, 'q_a_mvar'] = action[i,0] 
            self.network.asymmetric_sgen.at[i, 'q_b_mvar'] = action[i,1] 
            self.network.asymmetric_sgen.at[i, 'q_c_mvar'] = action[i,2] 

        pp.runpp_3ph(self.network)
        
        state_a = self.network.res_bus_3ph.iloc[self.injection_bus].vm_a_pu.to_numpy().reshape(-1,1)
        state_b = self.network.res_bus_3ph.iloc[self.injection_bus].vm_b_pu.to_numpy().reshape(-1,1)
        state_c = self.network.res_bus_3ph.iloc[self.injection_bus].vm_c_pu.to_numpy().reshape(-1,1)
        self.state = np.hstack([state_a, state_b, state_c])
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        if done:
            print('successful!')
        return self.state, reward, reward_sep, done

    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        # senario = 0
        sgen_num = len(self.network.asymmetric_sgen)
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.asymmetric_sgen['p_a_mw'] = 0.0
            self.network.asymmetric_sgen['q_a_mvar'] = 0.0
            self.network.asymmetric_sgen['p_b_mw'] = 0.0
            self.network.asymmetric_sgen['q_b_mvar'] = 0.0
            self.network.asymmetric_sgen['p_c_mw'] = 0.0
            self.network.asymmetric_sgen['q_c_mvar'] = 0.0
            
            # for i in range(sgen_num):
            for i in range(sgen_num):
                self.network.asymmetric_sgen.at[i, 'p_a_mw'] = -np.random.uniform(0.0, 0.08)
                self.network.asymmetric_sgen.at[i, 'p_b_mw'] = -np.random.uniform(0.0, 0.08)
                self.network.asymmetric_sgen.at[i, 'p_c_mw'] = -np.random.uniform(0.0, 0.08)  
            # self.network.asymmetric_sgen.at[0, 'p_a_mw'] = -np.random.uniform(0.05, 0.15)
            # self.network.asymmetric_sgen.at[0, 'p_b_mw'] = -np.random.uniform(0.05, 0.15)
            # self.network.asymmetric_sgen.at[0, 'p_c_mw'] = -np.random.uniform(0.05, 0.18)

            # self.network.asymmetric_sgen.at[1, 'p_a_mw'] = -np.random.uniform(0.00, 0.06)
            # self.network.asymmetric_sgen.at[1, 'p_b_mw'] = -np.random.uniform(0.00, 0.05)
            # self.network.asymmetric_sgen.at[1, 'p_c_mw'] = -np.random.uniform(0.00, 0.05)

            # self.network.asymmetric_sgen.at[2, 'p_a_mw'] = -np.random.uniform(0.0, 0.05)
            # self.network.asymmetric_sgen.at[2, 'p_b_mw'] = -np.random.uniform(0.0, 0.05)
            # self.network.asymmetric_sgen.at[2, 'p_c_mw'] = -np.random.uniform(0.01, 0.05)

            # self.network.asymmetric_sgen.at[3, 'p_a_mw'] = -np.random.uniform(0.0, 0.3)
            # self.network.asymmetric_sgen.at[3, 'p_b_mw'] = -np.random.uniform(0.0, 0.3)
            # self.network.asymmetric_sgen.at[3, 'p_c_mw'] = -np.random.uniform(0.0, 0.3)


        elif(senario == 1): #high voltage 
            self.network.asymmetric_sgen['p_a_mw'] = 0.0
            self.network.asymmetric_sgen['q_a_mvar'] = 0.0
            self.network.asymmetric_sgen['p_b_mw'] = 0.0
            self.network.asymmetric_sgen['q_b_mvar'] = 0.0
            self.network.asymmetric_sgen['p_c_mw'] = 0.0
            self.network.asymmetric_sgen['q_c_mvar'] = 0.0
            
            for i in range(sgen_num):
                self.network.asymmetric_sgen.at[i, 'p_a_mw'] = np.random.uniform(0.0, 0.1)
                self.network.asymmetric_sgen.at[i, 'p_b_mw'] = np.random.uniform(0.0, 0.1)
                self.network.asymmetric_sgen.at[i, 'p_c_mw'] = np.random.uniform(0.0, 0.1)   
        
        pp.runpp_3ph(self.network)

        state_a = self.network.res_bus_3ph.iloc[self.injection_bus].vm_a_pu.to_numpy().reshape(-1,1)
        state_b = self.network.res_bus_3ph.iloc[self.injection_bus].vm_b_pu.to_numpy().reshape(-1,1)
        state_c = self.network.res_bus_3ph.iloc[self.injection_bus].vm_c_pu.to_numpy().reshape(-1,1)
        self.state = np.hstack([state_a, state_b, state_c]) #shape: number_of_bus*3
        # print(state_a)
        return self.state

def create_eu_lv():
    pp_net = pp.networks.ieee_european_lv_asymmetric('on_peak_566') #off_peak_1440 off_peak_1 on_peak_566
    pp_net.asymmetric_sgen['p_a_mw'] = 0.0
    pp_net.asymmetric_sgen['q_a_mvar'] = 0.0
    pp_net.asymmetric_sgen['p_b_mw'] = 0.0
    pp_net.asymmetric_sgen['q_b_mvar'] = 0.0
    pp_net.asymmetric_sgen['p_c_mw'] = 0.0
    pp_net.asymmetric_sgen['q_c_mvar'] = 0.0

    # pp.create_asymmetric_sgen(pp_net, 25)
    pp.create_asymmetric_sgen(pp_net, 171)
    pp.create_asymmetric_sgen(pp_net, 133)
    pp.create_asymmetric_sgen(pp_net, 378)
    # pp.create_asymmetric_sgen(pp_net, 432)
    # pp.create_asymmetric_sgen(pp_net, 115)
    # pp.create_asymmetric_sgen(pp_net, 65)
    # pp.create_asymmetric_sgen(pp_net, 142)
    # pp.create_asymmetric_sgen(pp_net, 178)
    # pp.create_asymmetric_sgen(pp_net, 180)
    # pp.create_asymmetric_sgen(pp_net, 208)

    return pp_net

if __name__ == "__main__":
    net = create_eu_lv()
    injection_bus = np.array([171,133,378])
    env = Three_Phase_EU(net, injection_bus)
    state_list = []
    for i in range(100):
        state = env.reset(i)
        state_list.append(state)
    state_list = np.array(state_list)
    # print(state_list.shape)
    fig, axs = plt.subplots(3, 3, figsize=(9,9))
    for i in range(3):
        for j in range(len(injection_bus)):
            axs[i,j].hist(state_list[:,j,i])
    plt.show()
    
    