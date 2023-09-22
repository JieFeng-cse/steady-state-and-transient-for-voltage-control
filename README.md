# Bridging Transient and Steady-state Performance in Voltage Control: A Reinforcement Learning Approach With Safe Gradient Flow

This is the official repo for the paper titled as ''Bridging Transient and Steady-state Performance in Voltage Control: A Reinforcement Learning Approach With Safe Gradient Flow''.
## Abstract

Deep reinforcement learning approaches are becoming appealing for the design of nonlinear controllers for voltage control problems, but the lack of stability guarantees hinders their real-world deployment. This paper constructs a decentralized RL-based controller for inverter-based real-time voltage control in distribution systems. It features two components: a transient control policy and a steady-state performance optimizer. The transient policy is parameterized as a neural network,  and the steady-state optimizer represents the gradient of the long-term operating cost function. The two parts are synthesized through a safe gradient flow framework, which prevents the violation of reactive power capacity constraints. We prove that if the output of the transient controller is bounded and monotonically decreasing with respect to its input, then the closed-loop system is asymptotically stable and converges to the optimal steady-state solution. We demonstrate the effectiveness of our method by conducting experiments with IEEE 13-bus and 123-bus distribution system test feeders.

## Code Structure

This repo contains two tested simulation environments, the algorithm, training pipline, and the test cases. 

The algorithm is written in safeDDPG.py.
Plotting figures: test.py
Ploting the effect of alpha: effect_of_alph.py
Test stability condition: test_condition.py

Note: This code is for academic use only.

# How to train
>python train_DDPG.py --algorithm safe-ddpg --env_name 13bus --status train<br />
#customize your own algorithm, env_name and status<br />
#env: 13bus,123bus<br />
#algorithm: safe-ddpg<br />
#status: train,test<br />
#safe_method: safe-flow, no_gradient<br />
#use_safe_flow: Ture or False<br />
#use_gradient: Ture or False<br />
#check points are available<br />


## Citation
````
@ARTICLE{10163934,
  author={Feng, Jie and Cui, Wenqi and Cortés, Jorge and Shi, Yuanyuan},
  journal={IEEE Control Systems Letters}, 
  title={Bridging Transient and Steady-State Performance in Voltage Control: A Reinforcement Learning Approach With Safe Gradient Flow}, 
  year={2023},
  volume={7},
  number={},
  pages={2845-2850},
  doi={10.1109/LCSYS.2023.3289435}}

````