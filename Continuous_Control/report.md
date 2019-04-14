# Report

In this project, an agent is trained using Deep Reinforcement Learning to move a double-jointed arm and maintain it at its target position for as long as possible. The algorithm used is based on Deep Deterministic Policy Gradients (DDPG) algorithm described in the [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).

## Learning Algorithm

The DDPG algorithm is implemented using the following files using [this ddpg agent](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a starting point:

### [model.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Continuous_Control/model.py)
This file contains the classes for Actor and Critic - which are used to implement the agent as described in the paper mentioned above.

Both the actor and critic have 3 fully connected layers with RELU activations on the first two layers. The first layer of both the actor and critic has the size of the state vector and the last two layers have 400 and 300 units respectively. The number of possible actions determines the size of the output layer for the actor while the critic's output layer has a size of 1.

### [ddpg_agent.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Continuous_Control/ddpg_agent.py)
This file contains the agent that has an actor and a critic, both using a Q-Network local and Q-Network target that can interact with the environment and learn from the experiences stored in a replay buffer. Some noise is added with each action to let the agent explore the environment. Ornstein-Uhlenbeck process is used to add this noise which results in temporally correlated values centered around 0.

Learning from random samples, determined by BATCH_SIZE, from the experiences stored, determined by BUFFER_SIZE, helps avoid experience correlations while still learning from rare experiences.

GAMMA, the discount factor is set to 0.99 to let the agent prioritize long term rewards more than short term rewards. The agent learns 10 times for every 20 steps of interactions with the environment while collecting experiences at each step. LR, the learning rate determines the rate at which the agent learns.

As mentioned in the Benchmark Implementation section of the project, gradient clipping helped with the stability of the agent by clipping the norm of the gradients at 1 - `torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)`

### [Continuous_Control.ipynb](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Continuous_Control/Continuous_Control.ipynb)
This is the notebook where the agent is trained using the above files.

### Hyperparameters

The final parameters were very close to [this ddpg_agent](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py). I increased the buffer size to gather more experiences and increased the learning rate of the actor while decreasing the learning rate of the critic. I added a parameter EPSILON to gradually decrease the amount of noise added to the actions - controlling the rate of decrease using EPSILON_DECAY.

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 20       # how often to update the network
EPSILON = 1.0           # to control noise
EPSILON_DECAY = 1e-6    # to gradually decrease noise
```

## Rewards

The agent achieved an average score of 30 (over 100 episodes) in 133 episodes as can be seen [here](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Continuous_Control/Continuous_Control.ipynb).

```
Episode 20	Average Score: 0.87	Score: 1.09
Episode 40	Average Score: 1.34	Score: 3.79
Episode 60	Average Score: 1.99	Score: 4.94
Episode 80	Average Score: 3.01	Score: 10.32
Episode 100	Average Score: 4.15	Score: 7.07
Episode 120	Average Score: 6.71	Score: 20.36
Episode 140	Average Score: 10.51	Score: 21.24
Episode 160	Average Score: 15.07	Score: 27.24
Episode 180	Average Score: 19.92	Score: 29.07
Episode 200	Average Score: 24.40	Score: 21.39
Episode 220	Average Score: 28.16	Score: 32.50
Episode 233	Average Score: 30.14	Score: 36.77
Environment solved in 133 episodes!	Average Score: 30.14
```

Below is a plot of the rewards over episodes.

![rewards](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Continuous_Control/rewards.png)

## Ideas for future work

1. Use other algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) to solve the second version
2. Use Prioritized Experienced Replay
3. Fine tune parameters some more to see if the environment can be solved faster
4. Solve the crawler environment
