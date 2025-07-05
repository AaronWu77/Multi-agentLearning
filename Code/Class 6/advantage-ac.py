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
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

def train_actor_critic(num_episodes=1000, gamma=0.99, lr=1e-3, value_coeff=0.5):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(num_episodes):
        obs = env.reset()[0]
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        log_probs = []
        values = []
        rewards = []
        total_reward = 0

        done = False
        while not done:
            action_probs, state_value = model(obs)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            values.append(state_value.squeeze(0))  # Remove batch dim

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

        values = torch.stack(values)
        log_probs = torch.stack(log_probs)

        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).sum()
        critic_loss = value_coeff * (returns - values).pow(2).sum()

        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

# Run the training
if __name__ == "__main__":
    train_actor_critic()
