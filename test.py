import gymnasium as gym

# 本地环境中使用human模式
env = gym.make('CartPole-v1', render_mode='human')
observation, info = env.reset(seed=1)

for _ in range(100):
    env.render()  # 显示实时画面
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()