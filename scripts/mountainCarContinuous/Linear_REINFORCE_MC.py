import gym
import torch
import numpy as np
import matplotlib.pyplot as plt


class MC_REINFORCE():
    def __init__(self, env):
        super(MC_REINFORCE, self).__init__()
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.sigma = 0.1

        self._env = env
        self.model = np.zeros(env.observation_space.shape)
        self.data = []

    def cal_mu(self, state):
        return state.dot(self.model)

    # Gaussian policy
    def get_action(self, state):
        mu = self.cal_mu(state)
        action = np.random.normal(mu, self.sigma, 1)
        action = np.clip(action, self._env.action_space.low[0], self._env.action_space.high[0]) # make action inside of the range

        return action

    def update(self):
        G_t = 0
        for val in self.data[::-1]:
            state = val[0]
            action = val[1]
            reward = val[2]
            next_state = val[3]
            done = val[4]

            G_t = reward + self.gamma * G_t
            dlog_policy = (action - self.cal_mu(state)) * state / (self.sigma**2)
            self.model += self.learning_rate * dlog_policy * G_t

        self.data = []


def start_training(num_episode):
    env = gym.make('MountainCarContinuous-v0')
    PG_model = MC_REINFORCE(env)
    rewards = []

    for i_episode in range(num_episode):
        state = env.reset()
        done = False
        tot_reward = 0.0

        while not done:
            action = PG_model.get_action(state)
            next_state, reward, done, info = env.step(action)

            # env.render()

            PG_model.data.append((state, action, reward, next_state, done))
            tot_reward += reward

            state = next_state

        rewards.append(tot_reward)
        PG_model.update()


        if (i_episode+1) % 20 == 0:
            print(f'{i_episode+1} episodes done!')
            # print(PG_model.model)

    fig = plt.subplots()
    plt.plot(np.arange(num_episode), rewards)
    plt.show()

    return PG_model


if __name__=='__main__':
    PG_model = start_training(1000)

    print(f"training done with model {PG_model.model}")

    # env = gym.make('MountainCarContinuous-v0')
