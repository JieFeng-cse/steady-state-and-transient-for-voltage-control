from email import policy
# from multiprocessing.reduction import steal_handle
from turtle import forward
# from pyrsistent import T
from sqlalchemy import true
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import os
import random
import sys
import torch.optim as optim

# DPPG class
class DDPG:
    def __init__(self, policy_net, value_net,
                 target_policy_net, target_value_net,
                 value_lr=2e-4,
                 policy_lr=1e-4):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy_net = policy_net
        self.value_net = value_net
        self.target_policy_net = target_policy_net
        self.target_value_net = target_value_net
        
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        
        self.value_optimizer = optim.Adam(value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()

    def train_step(self, replay_buffer, batch_size,
                   gamma=0.99,
                   soft_tau=1e-2):

        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        next_action = action+self.target_policy_net(next_state, action)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()
        
        # linear_action = (torch.maximum(state-1.03,torch.zeros_like(state))-torch.maximum(0.97-state,torch.zeros_like(state)))*1
        # print(self.policy_net(state))
        # exit(0)
        # policy_loss = self.value_criterion(self.policy_net(state),linear_action)
        policy_loss = self.value_net(state, last_action+self.policy_net(state,last_action)) 
        policy_loss =  -policy_loss.mean()
        # policy_loss =  self.value_criterion(self.policy_net(state),linear_action)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    def train_step_3ph(self, replay_buffer, batch_size,
                   gamma=0.99,
                   soft_tau=1e-2):

        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        next_action = action-self.target_policy_net(next_state)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()

        jacob_list = []
        
        for i in range(state.shape[0]):
            x = state[i]
            x.requires_grad = True
            action_jb = self.policy_net(x)
            jacob = torch.zeros(state.shape[1], state.shape[1]) 
            for j in range(3):
                output = torch.zeros(state.shape[1]).to(self.device)
                output[j]=1
                jacob[j,:]=torch.autograd.grad(action_jb, x, grad_outputs=output, retain_graph=True)[0]    
            jacob_list.append(jacob.unsqueeze(0))
        jacob_list = torch.cat(jacob_list,0).to(self.device)          
        # print(torch.mean(jacob_list,dim=0))                                                                          
        jacob_ii = -torch.sum(jacob_list[:,0,0]+jacob_list[:,1,1]+jacob_list[:,2,2])
        jacob_dif = 0.
        for i in range(3):
            for j in range(3):
                if  i==j:
                    jacob_dif -= torch.sum(torch.abs(jacob_list[:,i,j]))
                else:
                    jacob_dif += torch.sum(torch.abs(jacob_list[:,i,j]))
        
        policy_loss = self.value_net(state, last_action-self.policy_net(state)) 
        policy_loss = -policy_loss.mean()+ torch.norm(self.policy_net(torch.ones_like(state)),2) + 0.05*torch.exp(jacob_ii) + 0.05*torch.exp(jacob_dif)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


# monotone policy network with dead-band between [v_min, v_max]
class SafePolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim,\
        up=1.0,low=-1.0, alpha=0.6, node_cost = 0.1,\
            use_gradient=True, safe_flow = True,\
            scale = 0.20, init_w=3e-3):
        super(SafePolicyNetwork, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        self.gradient_only = False
        self.env = env
        self.hidden_dim = hidden_dim
        self.scale = scale        
        
        #parameters for safe flow
        self.upper_bound_Q = up
        self.lower_bound_Q = low
        self.alpha = alpha
        self.node_cost = node_cost
        self.use_gradient = use_gradient
        self.use_safe_flow = safe_flow
        
        #define weight and bias recover matrix
        self.w_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.w_recover = -torch.triu(self.w_recover, diagonal=0)\
        +torch.triu(self.w_recover, diagonal=2)+2*torch.eye(self.hidden_dim)
        self.w_recover=self.w_recover.to(self.device)
        
        self.b_recover = torch.ones((self.hidden_dim, self.hidden_dim))
        self.b_recover = torch.triu(self.b_recover, diagonal=0)-torch.eye(self.hidden_dim)
        self.b_recover = self.b_recover.to(self.device)
        
        self.select_w = torch.ones(1, self.hidden_dim).to(self.device)
        self.select_wneg = -torch.ones(1, self.hidden_dim).to(self.device)
        
        # initialization
        self.b = torch.rand(self.hidden_dim)
        self.b = (self.b/torch.sum(self.b))*scale
        self.b = torch.nn.Parameter(self.b, requires_grad=True)
        
        self.c = torch.rand(self.hidden_dim)
        self.c = (self.c/torch.sum(self.c))*scale
        self.c = torch.nn.Parameter(self.c, requires_grad=True)
        
        self.q = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        self.z = torch.nn.Parameter(torch.rand(action_dim, self.hidden_dim), requires_grad=True)
        
    def forward(self, state, last_action):
        self.w_plus=torch.matmul(torch.square(self.q), self.w_recover)
        
        self.w_minus=torch.matmul(-torch.square(self.q), self.w_recover)
        
        b = self.b.data
        b = b.clamp(min=0)
        b = self.scale*b/torch.norm(b, 1)
        self.b.data = b
        
        #maybe this part is wrong?
        c = self.c.data
        c = c.clamp(min=0)
        c = self.scale*c/torch.norm(c, 1)
        self.c.data = c
        
        self.b_plus=torch.matmul(-self.b, self.b_recover) - torch.tensor(self.env.vmax-0.01)#-0.01
        self.b_minus=torch.matmul(-self.b, self.b_recover) + torch.tensor(self.env.vmin+0.01)#+0.01
        
        self.nonlinear_plus = torch.matmul(F.relu(torch.matmul(state, self.select_w)
                                                  + self.b_plus.view(1, self.hidden_dim)),
                                           torch.transpose(self.w_plus, 0, 1))
        
        self.nonlinear_minus = torch.matmul(F.relu(torch.matmul(state, self.select_wneg)
                                                   + self.b_minus.view(1, self.hidden_dim)),
                                            torch.transpose(self.w_minus, 0, 1))
        if self.use_gradient:
            # for high voltage scenario, the reactive power injection is negative
            x_high_voltage  = torch.tanh(self.nonlinear_plus)*\
                (torch.ones_like(last_action)*self.lower_bound_Q-last_action)*0.98*self.alpha
            # for low voltage scenario, the reactive power injection is positive
            x_low_voltage = -torch.tanh(self.nonlinear_minus)*\
                (torch.ones_like(last_action)*self.upper_bound_Q-last_action)*0.98*self.alpha
        else:
            # x_high_voltage = -self.nonlinear_plus
            # x_low_voltage = self.nonlinear_minus
            # for high voltage scenario, the reactive power injection is negative
            x_high_voltage  = torch.tanh(self.nonlinear_plus)*\
                (torch.ones_like(last_action)*self.lower_bound_Q-last_action)*self.alpha
            # for low voltage scenario, the reactive power injection is positive
            x_low_voltage = -torch.tanh(self.nonlinear_minus)*\
                (torch.ones_like(last_action)*self.upper_bound_Q-last_action)*self.alpha

        # x = (self.nonlinear_plus+self.nonlinear_minus) #previous version
        gradient = self.node_cost*last_action + torch.square(state) - torch.ones_like(state)
        if not self.use_gradient:
            gradient = 0

        x = x_high_voltage + x_low_voltage
        if gradient*x>0:
            x=0
        x -= gradient
        if self.use_gradient:
            x = self.safe_flow(x,last_action)
        return x
    
    def safe_flow(self, action,last_Q):
        action=torch.maximum(self.alpha*(self.lower_bound_Q-last_Q),action)
        action=torch.minimum(self.alpha*(self.upper_bound_Q-last_Q),action)
        return action

    def get_action(self, state, last_action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).to(self.device)
        action = self.forward(state, last_action)
        return action.detach().cpu().numpy()[0] 



class SafePolicy3phase(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, bus_id, scale = 0.15, init_w=3e-3):
        super(SafePolicy3phase,self).__init__()
        use_cuda = torch.cuda.is_available()
        self.env = env
        self.bus_id = bus_id
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        self.action_dim = action_dim
        action_dim_per_phase=1
        for phase in env.injection_bus[bus_id]:
            if phase == 'a':
                self.policy_a = SafePolicyNetwork(env, obs_dim, action_dim_per_phase, hidden_dim, scale = 0.15, init_w=3e-3)
            if phase == 'b':
                self.policy_b = SafePolicyNetwork(env, obs_dim, action_dim_per_phase, hidden_dim, scale = 0.15, init_w=3e-3)
            if phase == 'c':
                self.policy_c = SafePolicyNetwork(env, obs_dim, action_dim_per_phase, hidden_dim, scale = 0.15, init_w=3e-3)
    def forward(self, state):
        action_list = []
        for i,phase in enumerate(self.env.injection_bus[self.bus_id]):
            if phase == 'a':
                action = self.policy_a(state[:,i].unsqueeze(-1))
                action_list.append(action)
            if phase == 'b':
                action = self.policy_b(state[:,i].unsqueeze(-1))
                action_list.append(action)
            if phase == 'c':
                action = self.policy_c(state[:,i].unsqueeze(-1))
                action_list.append(action)
        action = torch.cat(action_list,dim=1)
        action += (torch.maximum(state-1.03, torch.zeros_like(state).to(self.device))-torch.maximum(0.97-state,  torch.zeros_like(state).to(self.device)))*0.01
        return action
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


# standard ddpg policy network
class PolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, \
        up=1.0,low=-1.0, alpha=0.6, node_cost = 0.1,\
            use_gradient=True, safe_flow = True,\
                 init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        #parameters for safe flow
        self.upper_bound_Q = up
        self.lower_bound_Q = low
        self.alpha = alpha
        self.node_cost = node_cost
        self.use_gradient = use_gradient
        self.use_safe_flow = safe_flow

        self.env = env
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, last_action=0):
        state.requires_grad = True
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        gradient = self.node_cost*last_action + torch.square(state) - torch.ones_like(state)
        if not self.use_gradient:
            gradient = 0
        x -= gradient
        x = self.safe_flow(x,last_action)
        return x
    
    def safe_flow(self, action,last_Q):
        action=torch.maximum(self.alpha*(self.lower_bound_Q-last_Q),action)
        action=torch.minimum(self.alpha*(self.upper_bound_Q-last_Q),action)
        return action

    def get_action(self, state, last_action=0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).to(self.device)
        action = self.forward(state,last_action)
        return action.detach().cpu().numpy()[0]

# linear
class LinearPolicy(nn.Module):
    def __init__(self, env, ph_num,\
        up=1.0,low=-1.0, alpha=0.6, node_cost = 0.1,\
            use_gradient=True, safe_flow = True):
        super(LinearPolicy, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        #parameters for safe flow
        self.upper_bound_Q = up
        self.lower_bound_Q = low
        self.alpha = alpha
        self.node_cost = node_cost
        self.use_gradient = use_gradient
        self.use_safe_flow = safe_flow

        self.env = env
        slope =torch.ones(1, requires_grad=True).to(self.device)
        self.slope = torch.nn.Parameter(slope)
        self.ph_num = ph_num

    def forward(self, state, last_action=0):
        state.requires_grad = True
        x_plus = torch.maximum(state-1.03, torch.zeros_like(state).to(self.device))*torch.square(self.slope)
        x_minus = torch.maximum(0.97-state, torch.zeros_like(state).to(self.device))*torch.square(self.slope)
        if self.use_safe_flow:
            # for high voltage scenario, the reactive power injection is negative
            x_high_voltage  = torch.tanh(x_plus)*\
                (torch.ones_like(last_action)*self.lower_bound_Q-last_action)*0.98*self.alpha
            # for low voltage scenario, the reactive power injection is positive
            x_low_voltage = torch.tanh(x_minus)*\
                (torch.ones_like(last_action)*self.upper_bound_Q-last_action)*0.98*self.alpha
        else:
            x_high_voltage = -x_plus
            x_low_voltage = x_minus

        # x = (self.nonlinear_plus+self.nonlinear_minus) #previous version
        gradient = self.node_cost*last_action + torch.square(state) - torch.ones_like(state)
        if not self.use_gradient:
            gradient = 0

        x = (x_high_voltage + x_low_voltage)*0
        x -= gradient
        x = self.safe_flow(x,last_action)
        return x
    
    def safe_flow(self, action,last_Q):
        action=torch.maximum(self.alpha*(self.lower_bound_Q-last_Q),action)
        action=torch.minimum(self.alpha*(self.upper_bound_Q-last_Q),action)
        return action

    def get_action(self, state, last_action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).to(self.device)
        action = self.forward(state, last_action)
        return action.detach().cpu().numpy()[0] 


# value network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ReplayBufferPI:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, last_action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

