import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")


class Policy(nn.Module):
    def __init__(self, learning_rate=0.001, gamma=0.9):
        super(Policy, self).__init__()
        self.data = []
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=0)
        )
        # self.fc1 = nn.Linear(4, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        output = self.model(x)
        # x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=0)
        return output

    def put_data(self, item):
        self.data.append(item)

    def train_net(self, device):
        G_t = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            G_t = r + self.gamma * G_t
            loss = -torch.log(prob).to(device) * G_t
            loss.backward()
        self.optimizer.step()
        self.data = []


def main():
    fig = plt.subplots()

    env = gym.make('CartPole-v1')
    pi = Policy().to(device)
    print_interval = 20
    score = 0.0
    rewards = []

    num_episode = 1000
    for i_episode in range(num_episode):
        state = env.reset()
        tot_reward = 0
        done = False

        while not done:
            prob = pi(torch.from_numpy(state).float().to(device)) # forward, probability
            m = Categorical(prob)
            a = m.sample()
            next_state, reward, done, info = env.step(a.item())
            # env.render()
            pi.put_data((reward, prob[a]))
            tot_reward += reward
            score += reward
            state = next_state

        pi.train_net(device)
        rewards.append(tot_reward)

        if (i_episode+1) % print_interval == 0:
            print(f"# of episode: {i_episode+1}, avg_reward: {score/print_interval}")
            score = 0.0

    env.close()

    plt.plot(np.arange(num_episode), rewards)
    plt.show()


if __name__=='__main__':
    main()