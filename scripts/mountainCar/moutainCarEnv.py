import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

print(env.action_space)
print(env.action_space.sample())

print(env.observation_space)
print(env.observation_space.sample())
print(env.observation_space.shape)
# [position, velocity] # (2, 1)

a = np.zeros((env.action_space.n, env.observation_space.shape[0]))
print(a.shape) # (3, 2)

state = env.observation_space.sample()
weight = np.zeros((env.action_space.n, env.observation_space.shape[0]))

action = env.action_space.sample()
Q = state.dot(weight[action])
print(Q.shape, Q)
