import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time


class linearVFA_MCcontrol_cartpole():
    def __init__(self):
        self._env = gym.make('CartPole-v1')
        self.que = deque([])
        self.weight = np.zeros((self._env.action_space.n, self._env.observation_space.shape[0]))
        self.epsilon = 0.01
        self.learning_rate = 0.01
        self.gamma = 0.9

    def get_action(self, state):
        nA = self._env.action_space.n
        A = np.ones(nA, dtype=float) * self.epsilon / nA
        max_action = np.argmax([self.Q(state, a) for a in range(nA)])
        A[max_action] += (1.0 - self.epsilon)
        sample = np.random.choice(nA, p=A)

        return sample

    def Q(self, state, action):
        value = state.dot(self.weight[action])

        return value

    def update(self):
        G_t = 0
        state = self.que[0][0]
        action = self.que[0][1]

        self.que.reverse()
        for sample in self.que:
            reward = sample[2]
            G_t = reward + self.gamma * G_t

        q_t = self.Q(state, action)

        delta = self.learning_rate * (G_t - q_t) * state
        self.weight[action] += delta

        self.que.clear()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = linearVFA_MCcontrol_cartpole()
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