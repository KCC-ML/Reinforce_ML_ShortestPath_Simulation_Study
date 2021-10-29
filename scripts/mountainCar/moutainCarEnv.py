import gym

env = gym.make('MountainCar-v0')
env.reset()

# for _ in range(1000):
#     env.step(2)
#     env.render()
#
# env.close()

print(env.action_space.sample())
print(env.observation_space)
print(env.observation_space.sample())
# [position, velocity]
print(env.observation_space.shape)

print(env.action_space)
print(env.action_space)