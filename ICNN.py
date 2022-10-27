from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np


class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""

    def __init__(self, dim=1, dimh=100, num_hidden_layers=4):
        super().__init__()

        Wzs = []
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(torch.nn.Linear(dimh, dimh, bias=False))
        Wzs.append(torch.nn.Linear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)

class convex_monotone_network(torch.nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, num_hidden_layers=4):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.icnn = ICNN(obs_dim ,hidden_dim,num_hidden_layers)

    def forward(self,state):
        # y = self.icnn(state)
        action = torch.zeros_like(state)
        for i in range(state.shape[0]):
            action[i] = torch.autograd.functional.jacobian(self.icnn,state[i].unsqueeze(0)-1,create_graph=True)
        # action.requires_grad = True 
        # print(action.shape,state.shape)
        assert action.shape[-1]==self.action_dim
        return action


    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()
