import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time


class linearVFA_MCcontrol_cartpole():
    def __init__(self, episodes=100):
        self.env = gym.make('CartPole-v1')
        self.episodes = episodes
        self.que = deque([])
        self.weight = np.zeros((self.env.action_space.n, self.env.observation_space.shape[0]))
        self.epsilon = 0.1
        self.learning_rate = 0.01
        self.gamma = 0.9

    def policy_evaluation(self):
        plt.ion()
        figure, ax = plt.subplots()

        rewards = []
        for e in range(1, self.episodes+1):
            state = self.env.reset()
            tot_reward = 0

            while True:
                action = self.policy(state, self.weight)
                next_state, reward, done, info = self.env.step(action)

                self.que.append([state, action, reward])

                tot_reward += reward
                state = next_state

                if done:
                    rewards.append(tot_reward)
                    self.update()
                    break

            if e % 100 == 0:
                print('{} episodes are done.'.format(e))

            x = np.arange(e)
            y = rewards
            line1, = ax.plot(x, y, color='blue')
            figure.canvas.draw()
            figure.canvas.flush_events()
            time.sleep(0.1)

    def policy(self, state, weight):
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) * self.epsilon / nA
        max_action = np.argmax([self.Q(state, a, weight) for a in range(nA)])
        A[max_action] += (1.0 - self.epsilon)
        sample = np.random.choice(nA, p=A)

        return sample

    def Q(self, state, action, weight):
        value = state.dot(weight[action])

        return value

    def update(self):
        # self.que.reverse()
        #
        # G_t = 0
        # for sample in self.que:
        #     state = sample[0]
        #     action = sample[1]
        #     reward = sample[2]
        #     G_t = reward + self.gamma * G_t
        #     q_t = self.Q(state, action, self.weight)
        #
        #     old = self.weight
        #     print(old)
        #     self.weight[action] += self.learning_rate * (G_t - q_t) * state
        #     new = self.weight
        #     print(new)
        #     conv_check = self.convergence_check(old, new)
        #
        # self.que.clear()

        G_t = 0
        state = self.que[0][0]
        action = self.que[0][1]

        self.que.reverse()
        for sample in self.que:
            reward = sample[2]
            G_t = reward + self.gamma * G_t

        q_t = self.Q(state, action, self.weight)

        delta = self.learning_rate * (G_t - q_t) * state
        self.weight[action] += delta

        self.que.clear()


if __name__ == "__main__":
    iteration = 1000
    agent = linearVFA_MCcontrol_cartpole(iteration)
    agent.policy_evaluation()

    print(agent.weight)

    rewards = []
    test_iter = 100
    for _ in range(test_iter):
        env = gym.make('CartPole-v1')
        state = env.reset()
        tot_reward = 0

        while True:
            action = agent.policy(state, agent.weight)
            state, reward, done, _ = env.step(action)
            tot_reward += reward
            # env.render()

            if done:
                rewards.append(tot_reward)
                break
        env.close()

    x = np.arange(test_iter)
    y = rewards

    plt.plot(x, y, color='blue')
    plt.show()