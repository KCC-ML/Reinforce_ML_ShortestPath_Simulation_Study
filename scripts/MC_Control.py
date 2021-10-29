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
        self.start_policy = [0.8, 0.1, 0.1]
        self.actions = self.translate_action_index(self.world.pacman_action_list)
        self.A = []
        self.greedy_A = []
        self.initialize_data()

        # MC-Control(On-policy)
        # T : episode 종료 시점 : int
        # pairs : t를 index로 갖고, pair<state, action> index를 element로 갖는 pair<state, action>의 집합 : vector(inf by 1)
        # pair : <state, action> : list(len : 4)
        # N : state counter, episode 중 나타났던 state 별로 횟수를 센다 : vector(state 수 by 1)
        # Gs : returns, 각 episode에서의 각 state에 대한 G 집합 : vector(state 수 by 1)
        # R : reward 집합, target index 외 모두 -1 : vector(state 수 by 1)
        # Q : 각 pair<state, action>에 대한 value 집합 : vector(state 수 * action 수 by 1)
        # policy : matrix[state 수][action 수]
        # A : probability of actions : matrix[state 수][action 수]
        # greedy_A : max probability actions among the actions(arg max_a Q(s,a)) : matrix[state 수][action 수]
        # 0으로 초기화 : N, Gs, Q

    def translate_action_index(self, actions):
        pacman_action_index = []
        for i, action in enumerate(actions):
            pacman_action_index.append(i)
        return pacman_action_index

    def initialize_data(self):
        # self.initialize_pairs()
        self.initialize_Q()
        self.initialize_N()
        self.initialize_R()
        self.initialize_Gs()
        self.matrixization_policy()

    def initialize_Q(self):
        self.Q = np.zeros((self.state_num * self.agent_direction_count, 1))

    def initialize_N(self):
        self.N = np.zeros((self.state_num, 1))

    def initialize_R(self):
        self.R = self.reward * np.ones((self.state_num, 1))
        temp = self.agent_direction_count * self.world.grid_dim * self.target_position[0] + self.agent_direction_count * \
               self.target_position[1]
        self.R[temp: temp + self.agent_direction_count] = 0

    def initialize_Gs(self):
        self.Gs = np.zeros((self.state_num, 1))

    def matrixization_policy(self):
        self.policy_matrix = np.reshape(self.start_policy * self.state_num, (self.state_num, len(self.start_policy)))
        temp = self.agent_direction_count * self.world.grid_dim * self.target_position[0] + self.agent_direction_count * self.target_position[1]
        self.policy_matrix[temp : temp + self.agent_direction_count, :] = 10

    def iteration(self):
        episode = 1
        while episode < 200:
            T, pairs = self.create_episode()   # episode 완료 -> T, pairs 확정
            G = 0
            for t in range(T-1, -1, -1):
                s_index = self.transform_state_index(pairs[t])
                G = self.gamma * G + self.R[s_index]
                if pairs[t] not in pairs[:t]:
                    self.Gs[s_index] += G
                    self.N[s_index] += 1
                    #print(self.Gs)
            episode += 1
            print(episode)
        for s in pairs:
            np.seterr(invalid='ignore')
            s_index = self.transform_state_index(s)
            self.Q[s_index] = self.Gs[s_index] / self.N[s_index]

        print(self.Q.reshape(-1, 4))

        time.sleep(5)
        self.policy_improvement()
        self.world.iter_step()

        return self.Q


    def create_episode(self):
        T, pairs = self.world.iter_step()
        #print(T, pairs)
        return T, pairs

    def transform_state_index(self, s):
        return s[0] * self.world.grid_dim * self.agent_direction_count + s[1] * self.agent_direction_count + s[2]


    def policy_improvement(self):
        for now_index in range(self.state_num):
            state_present = self.transform_index_state(now_index)
            if state_present[0:2] != list(self.target_position):
                self.policy_greedy_update(now_index, state_present)
                # print('policy_matrix : \n{}\n'.format(self.policy_matrix))
        print(self.policy_matrix.reshape(-1, 3))

    def policy_greedy_update(self, now_index, state_present):
        next_values = []
        for action in self.actions:
            next_index = self.next_state(state_present, action)
            next_values.append(self.Q[next_index])
        greedy_action = [i for i in range(len(next_values)) if next_values[i] == max(next_values)]
        self.policy_matrix[now_index, :] = 0
        self.policy_matrix[now_index, greedy_action] = 1 / len(greedy_action)

    def transform_index_state(self, state_index):
        row = state_index // (self.agent_direction_count * self.world.grid_dim)
        col = state_index % (self.agent_direction_count * self.world.grid_dim) // self.agent_direction_count
        direction = state_index % (self.agent_direction_count * self.world.grid_dim) % self.agent_direction_count
        return [row, col, direction]


    def next_state(self, state_present, action):
        state_next = copy.deepcopy(state_present)
        if action == 0:
            if self.world.env.walls[state_present[2]][state_present[0]][state_present[1]] == 1:
                pass
            else:
                if state_present[2] == 0:
                    state_next[0] -= 1
                elif state_present[2] == 1:
                    state_next[1] += 1
                elif state_present[2] == 2:
                    state_next[0] += 1
                elif state_present[2] == 3:
                    state_next[1] -= 1
        elif action == 1:
            state_next[2] = (state_next[2] - 1) % self.agent_direction_count
        elif action == 2:
            state_next[2] = (state_next[2] + 1) % self.agent_direction_count

        next_index = state_next[0] * (self.agent_direction_count * self.world.grid_dim) + state_next[1] * self.agent_direction_count + state_next[2]
        return next_index
