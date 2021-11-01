import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class linearVFA_MCcontrol_cartpole():
    def __init__(self, episodes=100):
        self.env = gym.make('CartPole-v1')
        self.episodes = episodes
        self.que = deque([])
        self.weight = np.zeros((self.env.action_space.n, self.env.observation_space.shape[0]))
        self.epsilon = 0.1
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.delta = 0.0001

    def policy_evaluation(self):
        for e in range(self.episodes):
            state = self.env.reset()
            tot_reward = 0

            while True:
                action = self.policy(state, self.weight)
                next_state, reward, done, info = self.env.step(action)

                self.que.append([state, action, reward])

                tot_reward += reward
                state = next_state

                if done:
                    conv_check = self.update()
                    break

            if e % 100 == 0:
                print('{} episodes are done.'.format(e))
            if conv_check:
                print(f'Iteration stops at {e} episodes.')
                break

    def state_scaler(self, state):
        state_min = np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38])
        state_max = np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38])

        scaled_state = (state - state_min) / (state_max - state_min)

        return scaled_state

    def policy(self, state, weight):
        nA = self.env.action_space.n
        A = np.ones(nA, dtype=float) * self.epsilon / nA
        max_action = np.argmax([self.Q(state, a, weight) for a in range(nA)])
        A[max_action] += (1.0 - self.epsilon)
        sample = np.random.choice(nA, p=A)

        return sample

    def Q(self, state, action, weight):
        state = self.state_scaler(state)
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

        conv_check = self.convergence_check(delta)

        self.que.clear()

        return conv_check

    def convergence_check(self, delta):
        print("===",delta)
        res = np.all(delta < self.delta)

        return res


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
            env.render()

            if done:
                rewards.append(tot_reward)
                break
        env.close()

    x = np.arange(test_iter)
    y = rewards

    plt.plot(x, y, color='blue')
    plt.show()