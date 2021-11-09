import numpy as np
import math

import time
import copy

class TDLbda():
    def __init__(self, world):
        self.world = world
        self.agent_direction_count = self.world.env.walls.shape[0]
        self.reward = -1
        self.target_reward = 0
        self.R = []
        self.gamma = 0.1
        self.state_num = self.world.grid_dim ** 2 * self.agent_direction_count
        self.target_position = self.world.env.gridmap_goal
        self.actions = self.translate_action_index(self.world.pacman_action_list)
        self.epsilon = 0.2
        self.start_policy = [1 / len(self.actions), 1 / len(self.actions), 1 / len(self.actions)]
        self.alpha = 0.4
        self.lbda = 0.1
        self.S = []
        self.initialize_data()

    def translate_action_index(self, actions):
        pacman_action_index = []
        for i, action in enumerate(actions):
            pacman_action_index.append(i)
        return pacman_action_index

    def initialize_data(self):
        self.initialize_V()
        self.initialize_E()
        # self.initialize_R()
        self.matrixization_policy()

    def initialize_V(self):
        self.V = np.zeros((self.state_num, 1))

    def initialize_E(self):
        self.E = np.zeros((self.state_num, 1))

    # def initialize_R(self):
    #     self.R = self.reward * np.ones((self.state_num, 1))
    #     temp = self.agent_direction_count * self.world.grid_dim * len(self.actions) * self.target_position[0] + self.agent_direction_count * len(self.actions) * self.target_position[1]
    #     self.R[temp: temp + self.agent_direction_count * len(self.actions)] = 5

    def matrixization_policy(self):
        self.policy_matrix = np.reshape(self.start_policy * self.state_num, (self.state_num, len(self.start_policy)))
        temp = self.agent_direction_count * self.world.grid_dim * self.target_position[0] + self.agent_direction_count * self.target_position[1]
        self.policy_matrix[temp : temp + self.agent_direction_count, :] = 10

    def iteration(self):
        # init for visualize convergence graph
        self.steps = []
        past_episode_step = math.inf

        # iteration init
        episode = 0

        # iteration start
        while episode < 300:
            episode += 1
            print('\nepisode = ', episode)

            step = self.policy_evaluation()

            # for visualize convergence graph
            self.steps.append(step)
            if step < past_episode_step:
                min_step = step
                now_optimal_policy = self.policy_matrix

            past_episode_step = step

        print('min_step = ', min_step)
        print('optimal_policy = \n', now_optimal_policy.reshape(-1, 3))
        self.world.window.destroy()


    def policy_evaluation(self):
        # episode(evaluation) init
        self.world.pacman.position = self.world.pacman.first_position
        self.world.pacman.cardinal_point = 'north'
        s = np.append(self.world.pacman.position, self.world.pacman_cardinal_points.index(self.world.pacman.cardinal_point))
        step = 1
        T = math.inf
        t = 0
        past_E = 0

        # evaluation start
        while True:
            next_s, R = self.world.iter_step()

            self.E[self.transform_s_index(s)] = self.gamma * self.lbda * past_E + 1
            delta = R + self.gamma * self.V[self.transform_s_index(next_s)] - self.V[self.transform_s_index(s)]
            self.V[self.transform_s_index(s)] += self.alpha * delta * self.E[self.transform_s_index(s)]

            self.policy_improvement()

            # tau = t - self.n + 1 # tau : time whose state's estimate is being updated
            # if tau >= 0:
            #     tmp_g = 0
            #     for i in range(tau + 1, min((tau + self.n), T) + 1):
            #         tmp_g += (self.gamma ** (i - tau - 1)) * self.R[i - 1]
            #     G = tmp_g
            #     if (tau + self.n) < T:
            #         G = G + (self.gamma ** self.n) * self.V[self.transform_s_index(self.S[tau + self.n])]
            #     self.V[self.transform_s_index(self.S[tau])] += self.alpha * (G - self.V[self.transform_s_index(self.S[tau])])
            #     #print(self.S[tau])
            #
            # if tau == T - 1: break

            if (np.all(self.target_position == next_s[:2])):
                break

            s = next_s
            past_E = self.E[self.transform_s_index(s)]
            t += 1
            step += 1
        print('total_step = ', step)

        return step


    def policy_improvement(self):
        for now_index in range(self.state_num):
            state_present = self.transform_index_s(now_index)
            if state_present[0:2] != list(self.target_position):
                self.policy_update(now_index, state_present)
                # print('policy_matrix : \n{}\n'.format(self.policy_matrix))

    def policy_update(self, now_index, state_present):
        next_values = []
        for action in self.actions:
            next_index = self.next_state(state_present, action)
            next_values.append(self.V[next_index])
        greedy_actions = [i for i in range(len(next_values)) if next_values[i] == max(next_values)]
        self.policy_matrix[now_index, :] = self.epsilon / len(self.actions)
        self.policy_matrix[now_index, greedy_actions] = (1 - self.epsilon + len(greedy_actions) * self.epsilon / len(self.actions)) / len(greedy_actions)

    def transform_s_index(self, s):
        unit_row = self.agent_direction_count * self.world.grid_dim
        unit_col = self.agent_direction_count
        return s[0] * unit_row + s[1] * unit_col + s[2]

    def transform_index_s(self, s_index):
        unit_row = self.agent_direction_count * self.world.grid_dim
        unit_col = self.agent_direction_count
        row = s_index // unit_row
        col = s_index % unit_row // unit_col
        direction = s_index % unit_row % unit_col
        return [row, col, direction]


    def next_state(self, s_present, action):
        s_next = copy.deepcopy(s_present)
        if action == 0:
            if self.world.env.walls[s_present[2]][s_present[0]][s_present[1]] == 1:
                pass
            else:
                if s_present[2] == 0:
                    s_next[0] -= 1
                elif s_present[2] == 1:
                    s_next[1] += 1
                elif s_present[2] == 2:
                    s_next[0] += 1
                elif s_present[2] == 3:
                    s_next[1] -= 1
        elif action == 1:
            s_next[2] = (s_next[2] - 1) % self.agent_direction_count
        elif action == 2:
            s_next[2] = (s_next[2] + 1) % self.agent_direction_count

        next_index = s_next[0] * (self.agent_direction_count * self.world.grid_dim) + s_next[1] * self.agent_direction_count + s_next[2]
        return next_index
