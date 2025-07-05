import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import ale_py
gym.register_envs(ale_py)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP-based policy network
class MLPPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# REINFORCE training loop
def train_reinforce(num_episodes=1000, gamma=0.99, lr=1e-3):

    #env = gym.make("ALE/Freeway-v5", obs_type="ram")
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = MLPPolicy(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(num_episodes):
        obs = env.reset()[0]
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        log_probs = []
        rewards = []
        total_reward = 0

        done = False
        while not done:
            probs = policy(obs)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
            rewards.append(reward)

            obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy gradient loss
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

# Run training
if __name__ == "__main__":
    train_reinforce()
