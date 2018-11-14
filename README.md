# Deep Deterministic Policy Gradients (DDPG) for Continuous Control

Solving a continuous control  task of double-jointed arms using DDPG agents.

## Environment details
The objective of Unity's [Reacher environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) is to move the 20 double-jointed arms to reach and maintain a target location for as long as possible. 

**Action space:** For each of the 20 arms, the action space is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

**State space:** The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities for each of the 20 arms.

**Reward function:** The reward is given to each agent individually: A reward of +0.1 is given each step the agent's arm is in its individual target location.

**DDPG structure:** Similar to Google DeepMind's paper, ["Continuous Control with Deep Reinforcement Learning"](https://arxiv.org/abs/1509.02971), the adopted learning algorithm is a DDPG algorithm. DDPG is a model-free policy-based reinforcement learning algorithm where agents learn by observing state spaces with no prior knowledge of the environment. Learning improves by using policy gradient optimization.

DDPG is an Actor-Critic model: 
* The Actor is a policy-based algorithm with high variance, taking relatively long to converge
* The Critic is a value-based algorithm with high bias instead
In this approach, Actor and Critic work together to reach better convergence and performance.

*Actor model*
Neural network with 3 fully connected layers:
* Fully connected layer 1: with input = 33 (state spaces) and output = 400
* Fully connected layer 2: with input = 400 and output = 300
* Fully connected layer 3: with input = 300 and output = 4, (for each of the 4 actions)
Tanh is used in the final layer that maps states to actions. Batch normalization is used for mini batch training.

*Critic model*
* Fully connected layer 1: with input = 33 (state spaces) and output = 400
* Fully connected layer 2: with input = 404 (states and actions) and output = 300
* Fully connected layer 3: with input = 300 and output = 1, (maps states and actions to Q-values)

**Parameters used in the DDPG algorithm:**
* Replay buffer size: BUFFER_SIZE = int(1e6)
* Minibatch size: BATCH_SIZE = 128
* Discount factor: GAMMA = 0.99
* Soft update of target parameters: TAU = 1e-3
* Learning rate of the actor: LR_ACTOR = 1e-4
* Learning rate of the critic: LR_CRITIC = 3e-4
* L2 weight decay: WEIGHT_DECAY = 0.0001

## Installation Instruction

Python 3.6 is required. The program requires PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git  
cd deep-reinforcement-learning/python  
pip install .
```

Run the following to create drlnd kernel in ipython so that the right unity environment is loaded correctly:  


```python -m ipykernel install --user --name drlnd --display-name "drlnd"```

## Getting Started

Place <mark>report.ipynb</mark> in the folder <mark>p2_continuous-control/</mark> together with the following two files:

* ddpg_agent.py - contains the DDPG agents code. 
* model.py - contains the neural network class

The Unity Reacher environment can be downloaded from here: 

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)  
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)  
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)  
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)  

Choose the environment suitable for your machine. Unzipping will create another Reacher_xxxx folder. For example, if the Linux Reacher environment is downloaded, ```Reacher_Linux``` will be created. 

Run ```p2_continuous-control/report.ipynb```

Enter the right path for the Unity Reacher environment in report.ipynb. 

Run the remaining cell as ordered in ```report.ipynb``` to train the DDPG agents. 
