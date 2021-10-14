import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scripts.pacman_entity import *


class TDzero_prediction():
    def __init__(self, pacman, numEpisode):
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.alpha = 0.1
        self.memory = []
        self.numEpisode = numEpisode
        self.totReward = np.array([])

        self.pacman = pacman
        self.grid_dim = pacman.n
        self.numState = 4 * self.grid_dim ** 2
        self.value_table = np.zeros(self.numState)
        self.action_list = [0, 1, 2]  # ["straight", "left", "right"]

        self.start_TDzero(self.numEpisode)
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

    def update(self, state, reward, next_state):
        TD_target = reward + self.gamma * self.value_table[next_state]
        TD_error = TD_target - self.value_table[state]
        self.value_table[state] += self.alpha * TD_error

    def start_TDzero(self, num_episode):
        for episode in range(num_episode):
            state = self.pacman.reset()
            done = False
            step = 0

            while True:
                action = self.get_action(state)
                next_state, reward, done = self.pacman.step(state, action)

                step += 1

                # update
                self.update(state, reward, next_state)
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
    TemporalDifferenceZero_prediction = TDzero_prediction(pacman, 1000)
    # print(MonteCarlo_policy.optimal_policy())
    print(TemporalDifferenceZero_prediction.value_table.reshape(-1, 4))