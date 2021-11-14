import numpy as np
import time
import math
import copy
from collections import deque

class SARSALF:
    def __init__(self, world):
        self.world = world
        self.agent_direction_count = self.world.env.walls.shape[0]
        self.reward = -1
        self.gamma = 0.1
        self.state_num = self.world.grid_dim ** 2 * self.agent_direction_count
        self.target_position = self.world.env.gridmap_goal
        self.actions = self.translate_action_index(self.world.pacman_action_list)
        self.epsilon = 0.1
        self.alpha = 0.5
        self.lbda = 0.9
        self.start_policy = [1 / len(self.actions), 1 / len(self.actions), 1 / len(self.actions)]
        self.greedy_A = []
        self.initialize_data()


    def translate_action_index(self, actions):
        pacman_action_index = []
        for i, action in enumerate(actions):
            pacman_action_index.append(i)
        return pacman_action_index

    def initialize_data(self):
        self.initialize_Q()
        # self.initialize_R()
        self.matrixization_policy()

    def initialize_Q(self):
        self.Q = np.zeros((self.state_num * len(self.actions), 1))

    def initialize_R(self):
        self.R = self.reward * np.zeros((self.state_num * len(self.actions), 1))
        # temp = self.agent_direction_count * self.world.grid_dim * len(self.actions) * self.target_position[0] + self.agent_direction_count * len(self.actions) * self.target_position[1]
        # self.R[temp: temp + self.agent_direction_count * len(self.actions)] = 5
        # print(self.R.size)
        # print(np.where(self.R == 0))

    def matrixization_policy(self):
        self.policy_matrix = np.reshape(self.start_policy * self.state_num, (self.state_num, len(self.start_policy)))
        # temp = self.agent_direction_count * self.world.grid_dim * self.target_position[0] + self.agent_direction_count * self.target_position[1]
        # #self.policy_matrix[temp : temp + self.agent_direction_count, :] = 10

    def iteration(self):
        self.steps = []
        episode = 0
        past_episode_step = math.inf
        while episode < 1000:
            episode += 1
            print('\nepisode = ', episode)
            step = self.policy_evaluation()
            self.policy_improvement()

            self.steps.append(step)
            if step < past_episode_step:
                min_step = step
                now_optimal_policy = self.policy_matrix

            past_episode_step = step

        print('min_step = ', min_step)
        print('optimal_policy = \n', now_optimal_policy.reshape(-1, 3))
        self.world.window.destroy()

    def policy_evaluation(self):
        self.world.pacman.position = self.world.pacman.first_position
        self.world.pacman.cardinal_point = 'north'
        cardinal_point_index = self.world.pacman_cardinal_points.index(self.world.pacman.cardinal_point)
        pacman_direction = np.random.choice(self.world.pacman_action_list, 1, p=self.policy_matrix[
            self.world.pacman.position[0] * (4 * 5) + self.world.pacman.position[1] * 4 + cardinal_point_index])
        pacman_direction_index = self.world.pacman_action_list.index(pacman_direction)
        pair = np.append(self.world.pacman.position, cardinal_point_index)
        pair = np.append(pair, pacman_direction_index)
        t = 1
        T = math.inf
        pairs = deque()
        pairs.append(pair)
        R = deque()
        R.append(-1)
        while True:
            if t < T:
                next_pair, reward = self.world.iter_step(pacman_direction)
                pacman_direction = np.random.choice(self.world.pacman_action_list, 1, p=self.policy_matrix[
                self.world.pacman.position[0] * (4 * 5) + self.world.pacman.position[
                    1] * 4 + self.world.pacman_cardinal_points.index(self.world.pacman.cardinal_point)])
                pacman_direction_index = self.world.pacman_action_list.index(pacman_direction)
                next_pair = np.append(next_pair, pacman_direction_index)
                pairs.append(next_pair)
                R.append(reward)
                if np.all(next_pair[:2] == self.target_position):
                    T = t + 1
                    break
                t += 1
        print('total_step = ', t)

        for t in range(T):
            lambda_G = 0
            for n in range(1, T):
                nstep_G = 0
                for i in range(1, n + 1):
                    nstep_G += (self.gamma ** (i - 1)) * R[i]
                lambda_G += (self.lbda ** n) * (
                            nstep_G + (self.gamma ** n) * self.Q[self.transform_pair_index(pairs[n])])
            lambda_G = (1 - self.lbda) * lambda_G + (self.lbda ** (T - 1)) * R[t]
            self.Q[self.transform_pair_index(pairs[0])] += self.alpha * (
                        lambda_G - self.Q[self.transform_pair_index(pairs[0])])

            pairs.popleft()
            R.popleft()
            T -= 1

        return t


    def policy_improvement(self):
        for s_index in range(self.state_num):
            s_action_values = self.Q[s_index * len(self.actions): s_index * len(self.actions) + 3]
            tmp = np.squeeze(s_action_values)
            self.greedy_A.append([])
            self.greedy_A[s_index] = np.where(tmp == np.max(tmp))
            for action_index in range(len(self.actions)):
                self.policy_matrix[s_index][action_index] = self.epsilon / len(self.actions)
                if np.any(action_index == self.greedy_A[s_index][0]):
                    self.policy_matrix[s_index][action_index] = (1 - self.epsilon + self.greedy_A[s_index][0].size * self.epsilon / len(self.actions)) / self.greedy_A[s_index][0].size


    def create_episode(self):
        T, pairs = self.world.iter_step()
        return T, pairs

    def transform_pair_index(self, pair):
        unit_row = self.agent_direction_count * self.world.grid_dim * len(self.actions)
        unit_col = self.agent_direction_count * len(self.actions)
        return pair[0] * unit_row + pair[1] * unit_col + pair[2] * len(self.actions) + pair[3]

    # def transform_index_pair(self, pair_index):
    #     unit_row = self.agent_direction_count * self.world.grid_dim * len(self.actions)
    #     unit_col = self.agent_direction_count * len(self.actions)
    #     row = pair_index // unit_row
    #     col = pair_index % unit_row // unit_col
    #     direction = pair_index % unit_row % unit_col // len(self.actions)
    #     action = pair_index % unit_row % unit_col % len(self.actions)
    #     return [row, col, direction, action]


    def next_pair(self, pair_present, action):
        pair_next = copy.deepcopy(pair_present)
        if action == 0:
            if self.world.env.walls[pair_present[2]][pair_present[0]][pair_present[1]] == 1:
                pass
            else:
                if pair_present[2] == 0:
                    pair_next[0] -= 1
                elif pair_present[2] == 1:
                    pair_next[1] += 1
                elif pair_present[2] == 2:
                    pair_next[0] += 1
                elif pair_present[2] == 3:
                    pair_next[1] -= 1
        elif action == 1:
            pair_next[2] = (pair_next[2] - 1) % self.agent_direction_count
        elif action == 2:
            pair_next[2] = (pair_next[2] + 1) % self.agent_direction_count

        next_index = pair_next[0] * (self.agent_direction_count * self.world.grid_dim) + pair_next[1] * self.agent_direction_count + pair_next[2]
        return next_index
