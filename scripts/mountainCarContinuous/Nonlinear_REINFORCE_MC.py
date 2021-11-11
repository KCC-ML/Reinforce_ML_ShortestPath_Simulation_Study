import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MC_REINFORCE_NONLINEAR(nn.Module):
    def __init__(self, env, learning_rate=0.001, gamma=0.9):
        super(MC_REINFORCE_NONLINEAR, self).__init__()
        self.data = []
        self.gamma = gamma
        self._env = env

        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc1(x))
        # x = self.fc2(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def put_data(self, data):
        self.data.append(data) # state, action, reward, next_state, done

    def train_net(self):
        G_t = 0
        self.optimizer.zero_grad()
        for value in self.data[::-1]:
            state = value[0]
            action = value[1]
            reward = value[2]
            next_state = value[3]
            done = value[4]

            G_t = reward + self.gamma * G_t
            mu, sigma = self.forward(torch.from_numpy(state).float())
            sigma = torch.abs(sigma)

            loss = -torch.log(torch.normal(mu, sigma))
            loss.backward()
        self.optimizer.step()
        self.data = []

    def get_action(self, state):
        with torch.no_grad():
            mu, sigma = self.forward(torch.from_numpy(state).float())
        action = np.random.normal(mu.numpy(), np.abs(sigma), 1)
        action = np.clip(action, self._env.action_space.low[0],
                         self._env.action_space.high[0])  # make action inside of the range

        return action


if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')
    agent = MC_REINFORCE_NONLINEAR(env)
    rewards = []

    num_epi = 1000
    for i_epi in range(num_epi):
        state = env.reset()
        done = False
        tot_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.put_data((state, action, reward, next_state, done))
            tot_reward += reward

            state = next_state

            # if (i_epi+1) % 20 == 0:
            #     env.render()

        agent.train_net()
        rewards.append(tot_reward)

        if (i_epi+1) % 20 == 0:
            print(f'{i_epi+1} episodes is done with reward {tot_reward}!')
            env.close()

    env.close()

    fig = plt.subplots()

    plt.plot(np.arange(num_epi), rewards)
    plt.show()


