import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import deque
from scripts.pacman_entity import *


class backward_TDl_prediction():
    def __init__(self, pacman, numEpisode, n):
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.alpha = 0.1
        self.lamda = 0.9
        self.numEpisode = numEpisode
        self.n_step = n
        self.que = deque([], maxlen=self.n_step)

        self.pacman = pacman
        self.grid_dim = pacman.n
        self.numState = 4 * self.grid_dim ** 2
        self.value_table = np.zeros(self.numState)
        self.eligibility = np.zeros(self.numState)
        self.action_list = [0, 1, 2]  # ["straight", "left", "right"]

        self.start_bTDl()
        # self.optimal_policy()

    def get_action(self, state):
        # if np.random.randn() < self.epsilon:
        #     idx = np.random.choice(len(self.action_list), 1)[0]
        # else:
        #     next_values = np.array([])
        #     for s in self.next_states(state):
        #         next_values = np.append(next_values, self.value_table[s])
        #     max_value = np.amax(next_values)
        #     tie_Qchecker = np.where(next_values == max_value)[0]
        #
        #     if len(tie_Qchecker) > 1:
        #         idx = np.random.choice(tie_Qchecker, 1)[0]
        #     else:
        #         idx = np.argmax(next_values)
        #
        # action = self.action_list[idx]
        idx = np.random.choice(len(self.action_list), 1, p=[0.8, 0.1, 0.1])[0]
        action = self.action_list[idx]

        return action

    def next_states(self, state):
        direction = state % 4
        row = (state // 4) // self.grid_dim
        col = (state // 4) % self.grid_dim

        next_states = []
        walls = self.pacman.walls
        for action in self.action_list:
            tmp = 0
            if action == 0:
                if walls[direction, row, col] == 1:
                    tmp = 0
                elif walls[direction, row, col] == 0:
                    if direction % 4 == 0:
                        tmp = -4 * self.grid_dim
                    elif direction % 4 == 1:
                        tmp = 4 * 1
                    elif direction % 4 == 2:
                        tmp = 4 * self.grid_dim
                    elif direction % 4 == 3:
                        tmp = -4 * 1
            elif action == 1:
                tmp = -direction + (direction - 1) % 4
            elif action == 2:
                tmp = -direction + (direction + 1) % 4

            next_states.append(state+tmp)

        return next_states

    def update(self, state, reward, next_state, done):
        self.que.append([state, reward, next_state])
        self.eligibility *= self.lamda * self.gamma
        self.eligibility[state] += 1.0

        if len(self.que) == self.n_step or done == True:
            TD_target = 0
            R_old = 0
            for i, tmp in enumerate(self.que):
                R_new = R_old + (self.gamma ** i) * tmp[1]
                V_new = (self.gamma ** (i+1)) * self.value_table[tmp[2]]

                TD_target += (R_new+V_new)

                R_old = R_new

            first_state = self.que[0][0]

            TD_error = TD_target - self.value_table[first_state]
            self.value_table[first_state] += self.alpha * TD_error * self.eligibility[first_state]

    def start_bTDl(self):
        for episode in range(self.numEpisode):
            state = self.pacman.reset()
            done = False
            step = 0

            while True:
                action = self.get_action(state)
                next_state, reward, done = self.pacman.step(state, action)

                step += 1

                self.update(state, reward, next_state, done)
                state = next_state

                if done:
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

        goal_state = 4 * ((self.pacman.n * self.pacman.gridmap_goal[0]) + self.pacman.gridmap_goal[1])
        for i in range(4):
            tmp = goal_state + i
            self.value_table[tmp] = self.pacman.goal_reward

    def optimal_policy(self):
        policy = np.zeros(self.numState)
        for state in range(self.numState):
            next_values = np.array([])
            for s in self.next_states(state):
                next_values = np.append(next_values, self.value_table[s])
            max_value = np.amax(next_values)
            tie_Qchecker = np.where(next_values == max_value)[0]

            if len(tie_Qchecker) > 1:
                idx = np.random.choice(tie_Qchecker, 1)[0]
            else:
                idx = np.argmax(next_values)
            policy[state] = self.action_list[idx]

        # print(policy.reshape(-1, 4))
        # print(self.value_table.reshape(-1, 4))
        return policy


if __name__ == "__main__":
    pacman = Pacman(5)
    TemporalDifference_prediction = backward_TDl_prediction(pacman, 1000, 1)
    # print(MonteCarlo_policy.optimal_policy())
    print(TemporalDifference_prediction.value_table.reshape(-1, 4))