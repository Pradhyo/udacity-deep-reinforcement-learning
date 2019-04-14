[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

This project is part of [Udacity's](https://www.udacity.com) [Deep  Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

An agent is trained using the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file.

### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!

### Files

- [Tennis.ipynb](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/Tennis.ipynb): Notebook where the agent is trained
- [model.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/model.py): Actor and Critic classes
- [ddpg_agent.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/ddpg_agent.py): Agent class than learns from the environment
- [multi_agent.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/multi_agent.py): Multi agent Class containing multiple ddpg agents that learns from the environment
- [checkpoint_actor_0.pth](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/checkpoint_actor_0.pth): Saved model weights of successful agent0's actor
- [checkpoint_actor_1.pth](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/checkpoint_actor_1.pth): Saved model weights of successful agent1's actor
- [checkpoint_critic__0.pth](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/checkpoint_critic_0.pth): Saved model weights of successful agent0's critic
- [checkpoint_critic_1.pth](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/checkpoint_critic_1.pth): Saved model weights of successful agent1's critic
- [report.md](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/report.md): Project report
