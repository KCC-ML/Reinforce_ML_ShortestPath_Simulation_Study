import numpy as np
import time

import copy

class MCControl():
    def __init__(self, world):
        self.world = world
        self.agent_direction_count = self.world.env.walls.shape[0]
        self.reward = -1
        self.gamma = 0.9
        self.state_num = self.world.grid_dim ** 2 * self.agent_direction_count
        self.target_position = self.world.env.gridmap_goal
        self.actions = self.translate_action_index(self.world.pacman_action_list)
        self.epsilon = 0.1
        self.start_policy = [1 / len(self.actions), 1 / len(self.actions), 1 / len(self.actions)]
        self.greedy_A = []
        self.initialize_data()

        # MC-Control(On-policy)
        # T : episode 종료 시점 : int
        # pairs : t를 index로 갖고, pair<state, action> index를 element로 갖는 pair<state, action>의 집합 : vector(inf by 1)
        # pair : <state, action> : list(len : 4)
        # N : pair counter, episode 중 나타났던 pair 별로 횟수를 센다 : vector(pair 수 by 1)
        # Gs : returns, 각 episode에서의 각 pair에 대한 G 집합 : vector(pair 수 by 1)
        # R : reward 집합, target index 외 모두 -1 : vector(pair 수 by 1)
        # Q : 각 pair<state, action>에 대한 value 집합 : vector(state 수 * action 수 by 1)
        # policy : matrix[state 수][action 수] (start_policy : epsilon-soft policy)
        # greedy_A : max probability actions among the actions(arg max_a Q(s,a)) : vector(state 수 by range(action 수))
        # 0으로 초기화 : N, Gs, Q

    def translate_action_index(self, actions):
        pacman_action_index = []
        for i, action in enumerate(actions):
            pacman_action_index.append(i)
        return pacman_action_index

    def initialize_data(self):
        self.initialize_Q()
        self.initialize_N()
        self.initialize_R()
        self.initialize_Gs()
        self.matrixization_policy()

    def initialize_Q(self):
        self.Q = np.zeros((self.state_num * len(self.actions), 1))

    def initialize_N(self):
        self.N = np.zeros((self.state_num * len(self.actions), 1))

    def initialize_R(self):
        self.R = self.reward * np.ones((self.state_num * len(self.actions), 1))
        temp = self.agent_direction_count * self.world.grid_dim * len(self.actions) * self.target_position[0] + self.agent_direction_count * len(self.actions) * self.target_position[1]
        self.R[temp: temp + self.agent_direction_count * len(self.actions)] = 0
        print(self.R.size)
        print(np.where(self.R == 0))

    def initialize_Gs(self):
        self.Gs = np.zeros((self.state_num * len(self.actions), 1))

    def matrixization_policy(self):
        self.policy_matrix = np.reshape(self.start_policy * self.state_num, (self.state_num, len(self.start_policy)))
        temp = self.agent_direction_count * self.world.grid_dim * self.target_position[0] + self.agent_direction_count * self.target_position[1]
        self.policy_matrix[temp : temp + self.agent_direction_count, :] = 10

    def iteration(self):
        episode = 1
        while episode < 3000:
            print('\nepisode = ', episode)
            T, pairs = self.create_episode()   # episode 완료 -> T, pairs 확정
            G = 0
            for t in range(T-1, -1, -1):
                pair_index = self.transform_pair_index(pairs[t])
                G = self.gamma * G + self.R[pair_index]
                if pairs[t].tolist() not in pairs[:t].tolist():
                    self.Gs[pair_index] += G
                    self.N[pair_index] += 1
                    self.Q[pair_index] = self.Gs[pair_index] / self.N[pair_index]
            for s_index in range(self.state_num):
                s_action_values = self.Q[s_index * len(self.actions): s_index * len(self.actions) + 3]
                tmp = np.squeeze(s_action_values)
                self.greedy_A.append([])
                self.greedy_A[s_index] = np.where(tmp == np.max(tmp))
                for action_index in range(len(self.actions)):
                    self.policy_matrix[s_index][action_index] = self.epsilon / len(self.actions)
                    if np.any(action_index == self.greedy_A[s_index][0]):
                        self.policy_matrix[s_index][action_index] = (1 - self.epsilon + self.greedy_A[s_index][0].size * self.epsilon / len(self.actions)) / self.greedy_A[s_index][0].size
            episode += 1

        time.sleep(5)
        self.world.iter_step()

        return self.Q


    def create_episode(self):
        T, pairs = self.world.iter_step()
        return T, pairs

    def transform_pair_index(self, pair):
        unit_row = self.agent_direction_count * self.world.grid_dim * len(self.actions)
        unit_col = self.agent_direction_count * len(self.actions)
        return pair[0][0] * unit_row + pair[0][1] * unit_col + pair[0][2] * len(self.actions) + pair[0][3]

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
