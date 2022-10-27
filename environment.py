import numpy as np
import gym
from gym import spaces
import os

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn
from pandapower.networks import ieee_european_lv_asymmetric
import torch
import matplotlib.pyplot as plt
from numpy import array, linalg as LA

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda:0" if use_cuda else "cpu")

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class VoltageCtrl_nonlinear(gym.Env):
    def __init__(self, pp_net, injection_bus, obs_dim=1, action_dim=1, 
                 v0=1, vmax=1.05, vmin=0.95): #1.05, 0.95
        
        self.network =  pp_net
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.action_space = spaces.Box(-100, 100, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-400.0, 400.0, (3,), dtype=np.float32)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.v0 = v0 
        self.vmax = vmax
        self.vmin = vmin
        
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])
 
        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(3*self.agentnum, )
        self.last_action = 0.0
    
    def step(self, action):
        
        done = False 
                    
        reward = float(-10*LA.norm(action)**2 -100*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))**2
                       - 100*LA.norm(np.clip(self.vmin-self.state, 0, np.inf))**2)   

        # action-transition dynamics         
        action = self.last_action + action
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.asymmetric_sgen.at[1, 'p_a_mw'] = action[i,0] 
            self.network.asymmetric_sgen.at[1, 'p_b_mw'] = action[i,1] 
            self.network.asymmetric_sgen.at[1, 'p_c_mw'] = action[i,2] 
        
        if not self.network['Converged_in_100_r']:
            print("Failed to converge in 100 rounds")
            print(action)
            reward -= 1000
            self.network['Converged_in_100_r'] = True
            done = True
        
        self.last_action = action

        pp.pf.runpp_3ph.runpp_3ph(self.network, algorithm='bfsw')
        
        self.state[0] = self.network.res_bus_3ph.iloc[self.injection_bus].vm_a_pu.to_numpy()
        self.state[1] = self.network.res_bus_3ph.iloc[self.injection_bus].vm_b_pu.to_numpy()
        self.state[2] = self.network.res_bus_3ph.iloc[self.injection_bus].vm_c_pu.to_numpy()
        
        if(np.min(self.state) > 0.9499 and np.max(self.state)< 1.0501):
            done = True
        
        return self.state, reward, done, {None:None}
    
    def reset(self, seed=1):
        seed = np.random.randint(0,200)
        # # seed = 51 #51 / 17
        np.random.seed(seed)
        self.last_action = 0.0
        # senario = np.random.choice([0, 1])
        senario = 0
        # print(senario)
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.asymmetric_sgen['p_a_mw'] = 0.0
            self.network.asymmetric_sgen['p_b_mw'] = 0.0
            self.network.asymmetric_sgen['p_c_mw'] = 0.0
            self.network.asymmetric_sgen['q_a_mvar'] = 0.0
            self.network.asymmetric_sgen['q_b_mvar'] = 0.0
            self.network.asymmetric_sgen['q_c_mvar'] = 0.0
            
            self.network.asymmetric_load['p_a_mw'] = 0.0
            self.network.asymmetric_load['p_b_mw'] = 0.0
            self.network.asymmetric_load['p_c_mw'] = 0.0
            self.network.asymmetric_load['q_a_mvar'] = 0.0
            self.network.asymmetric_load['q_b_mvar'] = 0.0
            self.network.asymmetric_load['q_c_mvar'] = 0.0
            
            self.network.asymmetric_sgen.at[1, 'p_a_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.asymmetric_sgen.at[1, 'p_b_mw'] = -0.5*np.random.uniform(2, 5)
            self.network.asymmetric_sgen.at[1, 'p_c_mw'] = -0.5*np.random.uniform(2, 5)
            
        
        pp.pf.runpp_3ph.runpp_3ph(self.network, algorithm='bfsw')
        self.state[0] = self.network.res_bus_3ph.iloc[self.injection_bus].vm_a_pu.to_numpy()
        self.state[1] = self.network.res_bus_3ph.iloc[self.injection_bus].vm_b_pu.to_numpy()
        self.state[2] = self.network.res_bus_3ph.iloc[self.injection_bus].vm_c_pu.to_numpy()
        return self.state

def create_56bus():
    pp_net = pp.converter.from_mpc('SCE_56bus.mat', casename_mpc_file='case_mpc')
    
    pp_net.asymmetric_sgen['p_a_mw'] = 0.0
    pp_net.asymmetric_sgen['p_b_mw'] = 0.0
    pp_net.asymmetric_sgen['p_c_mw'] = 0.0

    pp_net.asymmetric_sgen['q_a_mvar'] = 0.0
    pp_net.asymmetric_sgen['q_b_mvar'] = 0.0
    pp_net.asymmetric_sgen['q_c_mvar'] = 0.0

    pp.create_asymmetric_sgen(pp_net, 17, p_a_mw = 1.5, p_b_mw = 1.5, p_c_mw = 1.5, q_a_mvar=0, q_b_mvar=0, q_c_mvar=0)
    # pp.create_asymmetric_sgen(pp_net, 20, p_mw = 1, q_mvar=0)
    # pp.create_asymmetric_sgen(pp_net, 29, p_mw = 1, q_mvar=0)
    # pp.create_asymmetric_sgen(pp_net, 44, p_mw = 2, q_mvar=0)
    # pp.create_asymmetric_sgen(pp_net, 52, p_mw = 2, q_mvar=0)    
    return pp_net