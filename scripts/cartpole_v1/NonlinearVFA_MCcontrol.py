# Read commit first!
import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class NonlinearVFA_MCcontrol_cartpole(nn.Module):
    def __init__(self):
        self._env = gym.make('CartPole-v1')
        self.que = deque([])
        # self.weight = np.zeros((self._env.action_space.n, self._env.observation_space.shape[0]))
        self.epsilon = 0.01
        self.learning_rate = 0.01
        self.gamma = 0.9

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=0)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = NonlinearVFA_MCcontrol_cartpole()
    rewards = []

    num_epi = 1000
    for i_epi in range(num_epi):
        state = env.reset()
        tot_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.que.append([state, action, reward])
            tot_reward += reward
            # env.render()

            state = next_state

        rewards.append(tot_reward)
        agent.update()

        if (i_epi+1) % 20 == 0:
            print(f"Policy update with {i_epi+1} with reward {tot_reward}!")

    env.close()

    fig = plt.subplots()

    x = np.arange(num_epi)
    y = rewards

    plt.plot(x, y, color='blue')
    # plt.savefig('LinearVFA_MCcontrol_result_4.png')
    plt.show()