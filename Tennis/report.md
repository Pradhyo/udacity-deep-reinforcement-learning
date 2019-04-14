# Report

In this project, two agents are trained using Deep Reinforcement Learning to control tennis rackets and bounce a ball back and forth over a net for as long as possible. The algorithm used is based on Deep Deterministic Policy Gradients (DDPG) algorithm described in the [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

## Learning Algorithm

The DDPG algorithm is implemented using the following files using [this ddpg agent](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a starting point. Initially I used two separate DDPG agents with their own actors, critics and buffers but they couldn't learn enough to solve the environment. Creating a new multi-agent to let the agents share a replay buffer immediately produced much better results - I didn't even have to tune the hyperparameters that much for the agents to solve the environment.

### [model.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/model.py)
This file contains the classes for Actor and Critic - which are used to implement the agent as described in the paper mentioned above.

Both the actor and critic have 3 fully connected layers with RELU activations on the first two layers. The first layer of both the actor and critic has the size of the state vector and the last two layers have 256 and 128 units respectively. The number of possible actions determines the size of the output layer for the actor while the critic's output layer has a size of 1.

### [ddpg_agent.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/ddpg_agent.py)
This file contains the agent class that has an actor and a critic, both using a Q-Network local and Q-Network target that can interact with the environment and learn from experiences passed to them. Some noise is added with each action to let the agent explore the environment. Ornstein-Uhlenbeck process is used to add this noise which results in temporally correlated values centered around 0. The learning rates determine the rate at which the actor and critic of the agent learn.

As in the previous [Continuous Control](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/tree/master/Continuous_Control) project, gradient clipping helped with the stability of the agent by clipping the norm of the gradients at 1 - `torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)`

### [multi_agent.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/multi_agent.py)
This class creates two ddpg agents that share a replay buffer to act and learn from the environment using the act and learn functions of the individual agents as appropriate.

Learning from random samples, determined by BATCH_SIZE, from the experiences stored, determined by BUFFER_SIZE, helps avoid experience correlations while still learning from rare experiences. As both agents have the same goal, it made sense for them to learn from the same experiences stored in a shared buffer.

GAMMA, the discount factor is set to 0.99 to let the agent prioritize long term rewards more than short term rewards. The agent learns 5 times for each interaction with the environment while collecting experiences at each step.

### [Tennis.ipynb](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/Tennis.ipynb)
This is the notebook where the agent is trained using the above files.

### Hyperparameters

The final parameters were the same as [this ddpg_agent](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py). I added UPDATE_EVERY and UPDATE_TIMES to control how often the agent learns and for how many times.

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 1        # how often to update the network
UPDATE_TIMES = 5        # how many times to update the network
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

## Rewards

The agent achieved an average score of 0.5 (over 100 episodes) in 836 episodes as can be seen [here](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/Tennis.ipynb).

```
Episode 20	Average Score: 0.005	Score: 0.000
Episode 40	Average Score: 0.003	Score: 0.000
Episode 60	Average Score: 0.002	Score: 0.000
Episode 80	Average Score: 0.001	Score: 0.000
Episode 100	Average Score: 0.001	Score: 0.000
Episode 120	Average Score: 0.000	Score: 0.000
Episode 140	Average Score: 0.000	Score: 0.000
Episode 160	Average Score: 0.000	Score: 0.000
Episode 180	Average Score: 0.000	Score: 0.000
Episode 200	Average Score: 0.001	Score: 0.000
Episode 220	Average Score: 0.004	Score: 0.000
Episode 240	Average Score: 0.004	Score: 0.000
Episode 260	Average Score: 0.004	Score: 0.000
Episode 280	Average Score: 0.004	Score: 0.000
Episode 300	Average Score: 0.003	Score: 0.000
Episode 320	Average Score: 0.004	Score: 0.100
Episode 340	Average Score: 0.010	Score: 0.000
Episode 360	Average Score: 0.018	Score: 0.100
Episode 380	Average Score: 0.027	Score: 0.000
Episode 400	Average Score: 0.036	Score: 0.100
Episode 420	Average Score: 0.036	Score: 0.000
Episode 440	Average Score: 0.031	Score: 0.000
Episode 460	Average Score: 0.027	Score: 0.000
Episode 480	Average Score: 0.022	Score: 0.000
Episode 500	Average Score: 0.033	Score: 0.000
Episode 520	Average Score: 0.053	Score: 0.100
Episode 540	Average Score: 0.083	Score: 0.200
Episode 560	Average Score: 0.115	Score: 0.200
Episode 580	Average Score: 0.177	Score: 0.500
Episode 600	Average Score: 0.196	Score: 0.800
Checkpoint for average score 0.20 after 510 episodes!
Episode 620	Average Score: 0.208	Score: 0.100
Episode 640	Average Score: 0.208	Score: 0.300
Episode 660	Average Score: 0.206	Score: 0.200
Episode 680	Average Score: 0.187	Score: 0.200
Episode 700	Average Score: 0.201	Score: 0.100
Episode 720	Average Score: 0.229	Score: 0.200
Episode 740	Average Score: 0.247	Score: 0.300
Episode 760	Average Score: 0.290	Score: 1.800
Episode 780	Average Score: 0.275	Score: 0.000
Episode 800	Average Score: 0.295	Score: 0.100
Checkpoint for average score 0.31 after 703 episodes!
Episode 820	Average Score: 0.309	Score: 1.000
Episode 840	Average Score: 0.313	Score: 0.600
Episode 860	Average Score: 0.355	Score: 0.600
Checkpoint for average score 0.40 after 775 episodes!
Episode 880	Average Score: 0.427	Score: 1.200
Episode 900	Average Score: 0.429	Score: 0.600
Episode 920	Average Score: 0.463	Score: 0.100
Checkpoint for average score 0.50 after 836 episodes!
```

Below is a plot of the rewards over episodes.

![rewards](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Tennis/rewards.png)

## Ideas for future work

1. Use other algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb)
2. Use Prioritized Experienced Replay
3. Fine tune parameters some more to see if the environment can be solved faster
4. Try a shared critic along with the shared buffer
5. Solve the soccer environment
