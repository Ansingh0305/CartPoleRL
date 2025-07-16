# CartPole with Deep Q-Learning (DQN)

This project demonstrates how to solve the `CartPole-v1` environment from OpenAI Gym using a Deep Q-Network (DQN), a popular reinforcement learning algorithm.

## Key Features

*   **Deep Q-Network (DQN)**: A neural network defined in `cartpole_train.py` is used to approximate the Q-value function, which estimates the expected return for taking an action in a given state.
*   **Experience Replay**: The `ReplayBuffer` class stores transitions (state, action, reward, next_state, done) that the agent observes. During training, mini-batches are randomly sampled from this buffer to break the correlation between consecutive samples and improve learning stability.
*   **Target Network**: A separate `target_net` is used to calculate the target Q-values. Its weights are periodically updated with the weights from the main `policy_net`, which helps to stabilize the learning process.
*   **Epsilon-Greedy Strategy**: To balance exploration and exploitation, the agent selects a random action with a probability `epsilon` and the best-known action (according to the policy network) with a probability `1-epsilon`. The value of `epsilon` decays over time.
*   **Custom Reward Shaping**: The `custom_reward` function provides a more nuanced reward signal than the default environment reward. It heavily penalizes failure and rewards the agent for keeping the pole upright and centered.

## How Reinforcement Learning is Applied

In this project, a DQN agent learns to balance a pole on a cart.

*   **Agent**: The DQN model which controls the cart.
*   **Environment**: The `CartPole-v1` environment from `gym`.
*   **State**: A 4-dimensional vector representing the cart's position, velocity, the pole's angle, and its angular velocity.
*   **Action**: The agent can take one of two discrete actions: push the cart left or right.
*   **Reward**: The agent receives a reward at each step based on the `custom_reward` function. The goal is to learn a policy that maximizes the cumulative reward over an episode.

The agent learns by minimizing the Mean Squared Error between the predicted Q-values from the `policy_net` and the target Q-values calculated using the `target_net`.

## How to Use

### Prerequisites

Make sure you have Python and the following libraries installed:
```bash
pip install gym torch numpy
```

### 1. Train the Model

Run the training script. This will train the agent and save the learned model weights to `cartpole_dqn.pth`.

```bash
python cartpole_train.py
```

### 2. Run the Trained Agent

To watch the trained agent perform in the environment, run the evaluation script. This script loads the `cartpole_dqn.pth` file and renders the environment.

```bash
python cartpole.py
```
