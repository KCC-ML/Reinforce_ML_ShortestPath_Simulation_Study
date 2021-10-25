import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import deque
from scripts.pacman_entity import *


class bSARSAl():
    def __init__(self, pacman, numEpisode):
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.gamma = 0.9
        self.lamda = 0.9

        self.pacman = pacman
        self.grid_dim = pacman.n
        self.numState = 4 * self.grid_dim ** 2

        self.action_list = [0, 1, 2]  # ["straight", "left", "right"]
        self.eligibility = np.zeros((self.numState, len(self.action_list)))

        self.q_table = np.zeros((self.numState, len(self.action_list)))
        self.policy = np.zeros((self.numState, len(self.action_list)))

        self.policy_evaluation(numEpisode)

    def get_action(self, state):
        if np.random.randn() < self.epsilon:
            idx = np.random.choice(len(self.action_list), 1)[0]
        else:
            max_value = np.amax(self.q_table[state])
            tie_Qchecker = np.where(self.q_table[state] == max_value)[0]

            if len(tie_Qchecker) > 1:
                idx = np.random.choice(tie_Qchecker, 1)[0]
            else:
                idx = np.argmax(self.q_table[state])

        action = self.action_list[idx]
        return action

    def update(self, state, action, reward, next_state, next_action, done, episode):
        self.eligibility *= self.lamda * self.gamma
        self.eligibility[state][action] += 1.0

        Q_target = reward + self.gamma * self.q_table[next_state][next_action]
        Q_error = Q_target - self.q_table[state][action]

        # Q2?
        # Why update all state, action pair in each step of episode?
        # How to determine td_error for not visit state, action pair?
        self.q_table += self.learning_rate * Q_error * self.eligibility
        # self.q_table[self.que[0][0]][self.que[0][1]] += self.learning_rate * Q_error * self.eligibility[self.que[0][0]][self.que[0][1]]

        self.policy_improvement(episode)

    def policy_evaluation(self, num_episode):
        for episode in range(num_episode):
            self.eligibility = np.zeros((self.numState, len(self.action_list)))
            state = self.pacman.reset()
            action = self.get_action(state)
            done = False
            step = 0

            while True:
                next_state, reward, done = self.pacman.step(state, action)
                next_action = self.get_action(next_state)

                self.update(state, action, reward, next_state, next_action, done, episode)

                step += 1
                state = next_state
                action = next_action

                if done:
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

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
    pacman = Pacman(5)
    bSARSAl_policy = bSARSAl(pacman, 100)
    print(bSARSAl_policy.q_table.reshape(-1, 3))