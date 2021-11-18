import gym
import numpy as np

env = gym.make('MountainCarContinuous-v0')
obs = env.reset()
print(obs)

action = env.action_space.sample()
print(action)

obs_space = env.observation_space.shape
print(obs_space)

low = env.action_space.low[0]
high = env.action_space.high[0]
print(low, high)

a = np.random.normal(0, 0.1, 1)
a = np.clip(action[0], env.action_space.low[0], env.action_space.high)
print(a)


tmp = np.array([0, 2])
a, b = tmp
print(a, b)