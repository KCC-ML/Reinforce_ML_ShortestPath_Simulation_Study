import numpy as np

class MDP:
    def __init__(self, walls):
        # 불변값 : 사용자 지정값, 환경값 정의
        self.state_num = 100
        self.grid_dim = 5
        self.actions = [0, 1, 2]
        self.start_policy = [1.0, 0.8, 0.8]
        self.target_position = [3, 2]
        self.walls = walls
        self.reward = -1
        self.gamma = 0.1
        self.theta = 0.001
        self.initialize_elements()

    def initialize_elements(self):
        self.matrixization_value()
        self.matrixization_policy()
        self.matrixization_reward()

    def matrixization_value(self):
        self.value_matrix = np.zeros((self.state_num, 1))

    def matrixization_policy(self):
        self.policy_matrix = np.reshape(self.start_policy * self.state_num, (self.state_num, len(self.start_policy)))
        temp = 4 * 5 * self.target_position[0] + 4 * self.target_position[1]
        self.policy_matrix[temp : temp + 4, :] = 10

    def matrixization_reward(self):
        self.rewards = self.reward * np.ones((self.state_num, 1))
        temp = 4 * 5 * self.target_position[0] + 4 * self.target_position[1]
        self.rewards[temp : temp + 4] = 0

    def policy_iteration(self):
        iteration = 0
        self.policy_stable = False
        while not self.policy_stable:
            self.policy_evaluation()
            self.policy_improvement()
            iteration += 1
        print('iteration : {}\n'.format(iteration))
        print('greedy_policy_matrix : {}\n'.format(self.greedy_policy_matrix))
        print('greedy_qvalue_matrix : {}\n'.format(self.greedy_qvalue_matrix.reshape((-1,4))))
        return self.greedy_policy_matrix

    def policy_evaluation(self):
        episode = 0
        delta_qvalue = 0
        while delta_qvalue < self.theta:
            delta_qvalue = 0
            step = 0
            for now_index in range(self.state_num):
                state_present = self.transform_state_index(now_index)
                if state_present[0:2] == self.target_position:
                    qvalue_present = 0
                else:
                    qvalue_past = self.value_matrix[now_index]
                    qvalue_present = self.qvalue_update(now_index, state_present)
                delta_qvalue = max(delta_qvalue, abs(qvalue_past - qvalue_present))
                self.value_matrix[now_index] = qvalue_present
                # print('value_matrix : {}\n'.format(self.value_matrix.reshape((-1,4))))
                # print('delta_qvalue : {}\n'.format(delta_qvalue))
                step += 1
            print('step : {}\n'.format(step))
            episode += 1
            print('delta_qvalue : {}\n'.format(delta_qvalue))
        print('episode : {}\n'.format(episode))
        self.greedy_qvalue_matrix = self.value_matrix

    def policy_improvement(self):
        for now_index in range(self.state_num):
            state_present = self.transform_state_index(now_index)
            if state_present[0:2] != self.target_position:
                old_action = self.policy_matrix[now_index]
                self.policy_greedy_update(now_index, state_present)
                # print('policy_matrix : {}\n'.format(self.policy_matrix))
                if np.array_equal(old_action, self.policy_matrix[now_index]):
                    self.policy_stable = True
                    self.greedy_policy_matrix = self.policy_matrix

    def qvalue_update(self, now_index, state_present):
        sum = 0
        for action in self.actions:
            next_index = self.next_state(state_present, action)
            sum += self.policy_matrix[now_index][action] * (self.rewards[next_index] + self.gamma * self.value_matrix[next_index])
        qvalue_present = sum
        return qvalue_present

    def policy_greedy_update(self, now_index, state_present):
        next_values = []
        for action in self.actions:
            next_index = self.next_state(state_present, action)
            next_values.append(self.value_matrix[next_index])
        greedy_action = [i for i in range(len(next_values)) if next_values[i] == max(next_values)]
        self.policy_matrix[now_index, :] = 0
        self.policy_matrix[now_index, greedy_action] = 1 / len(greedy_action)

    def transform_state_index(self, state_index):
        row = state_index // (4 * 5)
        col = state_index % (4 * 5) // 4
        direction = state_index % (4 * 5) % 4
        return [row, col, direction]

    def next_state(self, state_present, action):
        state_next = state_present
        if action == 0:
            if self.walls[state_present[2]][state_present[0]][state_present[1]] == 1:
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
            state_next[2] = (state_next[2] - 1) % 4
        elif action == 2:
            state_next[2] = (state_next[2] + 1) % 4

        next_index = state_next[0] * (4 * 5) + state_next[1] * 4 + state_next[2]
        return next_index