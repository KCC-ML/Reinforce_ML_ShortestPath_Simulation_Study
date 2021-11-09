import gym
import numpy as np

env = gym.make('MountainCarContinuous-v0')
obs = env.reset()
print(obs)

action = env.action_space
print(action)

obs_space = env.observation_space.shape
print(obs_space)


a = np.random.normal(0, 0.1, 1)
print(a)