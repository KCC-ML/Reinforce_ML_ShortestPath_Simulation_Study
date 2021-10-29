import numpy as np
import gym
from collections import deque
import time

class linearSVFA():
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.linear_model = np.zeros(self.env.observation_space.shape)
        self.que = deque([])

        self.learning_rate = 0.01
        self.gamma = 0.9

    def update(self):
        self.que.reverse()

        G_t = 0
        for sample in self.que:
            state = sample[0]
            reward = sample[1]
            G_t = reward + self.gamma * G_t
            V_t = self.linearState(state)

            self.linear_model += self.learning_rate * (G_t - V_t) * state

        self.que.clear()

    def linearState(self, state):
        return np.dot(self.linear_model, state)

    def policy_evaluation(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while True:
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                self.que.append([state, reward, done])

                state = next_state

                if done:
                    self.update()
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

        return self.linear_model

    def policy_improvement(self, episode):
        self.epsilon = 1.0 / (episode + 1)

        for state in range(self.numState):
            max_value = np.amax(self.q_table[state])
            tie_Qchecker = np.where(self.q_table[state] == max_value)[0]

            if len(tie_Qchecker) > 1:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state, tie_Qchecker] = (1 - self.epsilon) / len(tie_Qchecker) + self.epsilon / len(
                    self.action_list)
            else:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state, tie_Qchecker] = 1 - self.epsilon + self.epsilon / len(self.action_list)


if __name__ == "__main__":
    coefficient = linearSVFA().policy_evaluation(10000)
    print(coefficient)

    env = gym.make('MountainCar-v0')
    env.reset()
    done = False
    max_position = -1.2
    while not done:
        next_values = np.array([])
        for action in range(3):
            next_state, reward, done, info = env.step(action)
            next_values = np.append(next_values, linearSVFA().linearState(next_state))

        max_value = np.amax(next_values)
        tie_Qchecker = np.where(next_values == max_value)[0]

        if len(tie_Qchecker) > 1:
            idx = np.random.choice(tie_Qchecker, 1)[0]
        else:
            idx = np.argmax(next_values)

        action = idx

        res = env.step(action)
        if res[0][0] > max_position:
            max_position = res[0][0]
        env.render()
        print(max_position)

    env.close()
