# Report

In this project, an agent is trained using Reinforcement Learning to navigate and collect yellow bananas while avoiding blue bananas in a large, square world. The algorithm used is based on Deep Q-Network (DQN) described in the [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).

## Learning Algorithm

The Deep Q-Network algorithm used in this project builds on Q-learning, where an agent learn an action-value function based on its interactions with an environment. For a given state the agent is in, the agent performs an action resulting in a new state and a corresponding reward for that action. Using all of this information, the agent updates its action-value function to eventually learn a policy that lets it maximize rewards over a long time. 

Q-learning is a type of Temporal-Difference method where the agent learns continuously without waiting for a final outcome and thus suitable for the continuous task at hand here. Since the states are also continuous in this project, neural networks are used as function approximators and hence the name *Deep* Q-Network. This however causes a lot of instability and these have been addressed using the below techniques.

1. **Experience Replay** - experiences (state, action, reward, new state) are stored in a buffer while interacting with the environment. Instead of learning from each of these in sequence, random samples from these experiences are used for learning. This breaks correlation as experiences next to each other are usually highly correlated. Another advantage is that the agent learns from rare experiences multiple times.
2. **Fixed Q-Targets** - the weights are updated during the learning process but this affects both the target and the Q value causing instability. To avoid updating a guess with a guess, we teach two similar networks. One of these will be fixed (target) while the local network learns and then updates the target network periodically so the agent controlled by the local network is more stable. 

The implementation of the algorithm described above is in the following files:

### [model.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Navigation/model.py)
This file contains the class for a Q-Network - which is used to implement the agent using a local and target network as described in the paper mentioned above.

The Q-Network has 3 fully connected layers with RELU activations on the first two layers. The first layer has the size of the state vector and the last two layers have 64 units. The number of possible actions determines the size of the output layer.

### [dqn_agent.py](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Navigation/dqn_agent.py)
This file contains the agent using a Q-Network local and Q-Network target that can interact with the environment and learn from the experiences stored in a replay buffer. Learning from random samples, determined by BATCH_SIZE, from the experiences stored, determined by BUFFER_SIZE, helps avoid experience correlations while still learning from rare experiences.

GAMMA, the discount factor is set to 0.99 to let the agent prioritize long term rewards more than short term rewards. The agent only learns every 4 steps of interactions with the environment while collecting experiences at each step. LR, the learning rate determines the rate at which the agent learns.

### [Navigation.ipynb](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Navigation/Navigation.ipynb)
This is the notebook where the agent is trained using the above files.

### Hyperparameters

I didn't have to tune the hyperparameters as the defaults from [this dqn_agent](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/dqn_agent.py) solved the environment comfortably.

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
```

## Rewards

The agent achieved an average score of 15 (over 100 episodes) in 605 episodes. An average score of 13 was reached before 500 episodes itself as can be seen [here](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Navigation/Navigation.ipynb).

```
Episode 100 Average Score: 0.96
Episode 200 Average Score: 4.45
Episode 300 Average Score: 8.49
Episode 400 Average Score: 11.12
Episode 500 Average Score: 12.31
Episode 600 Average Score: 13.72
Episode 700 Average Score: 14.84
Episode 705 Average Score: 15.00
Environment solved in 605 episodes! Average Score: 15.00
```

Below is a plot of the rewards over episodes.

![rewards](https://github.com/Pradhyo/udacity-deep-reinforcement-learning/blob/master/Navigation/rewards.png)

## Ideas for future work

1. Use Prioritized Experienced Replay
2. Dueling Deep Q Networks
3. Fine tune more parameters
4. Learn from pixels
