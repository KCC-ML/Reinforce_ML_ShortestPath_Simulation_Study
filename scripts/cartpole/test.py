import gym

env = gym.make('CartPole-v1')

print(env.action_space.n)
print(env.observation_space)