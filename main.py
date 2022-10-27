import argparse
import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
from tensorboardX import SummaryWriter
from environment import VoltageCtrl_nonlinear,create_56bus

import os
import gym
import numpy as np
from gym import wrappers

import torch
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp
import pandapower.networks as pn

pp_net = create_56bus()
injection_bus = np.array([17])
env = VoltageCtrl_nonlinear(pp_net, injection_bus)

for i in range(100):
    action = np.random.randn(1,3,1)*0.1
    print(action[0])
    next_state, reward, done, _ = env.step(action)


