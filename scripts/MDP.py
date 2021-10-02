import numpy as np
import copy

class MDP:
    def __init__(self, grid_dim, walls, gridmap_goal, actions):
        # 불변값 : 사용자 지정값, 환경값 정의
        self.state_num = grid_dim**2 * 4
        self.grid_dim = grid_dim
        self.actions = self.translate_action_index(actions)
        self.start_policy = [0.8, 0.1, 0.1]
        self.target_position = gridmap_goal
        self.walls = walls
        self.reward = -1
        self.gamma = 0.1
        self.theta = 0.01
        self.initialize_elements()

        # gamma = 1 -> value_function : 음의 방향으로 발산 -> policy evaluation 중단 기준 : episode 수
            # episode 수가 크면 iteration이 적게 필요하고
            # episode 수가 작으면 iteration이 많이 필요
            # 즉, policy improvement가 자주 될수록 episode 수가 작아도 됨
        # gamma = 소수 -> value_function : 음의 방향으로 수렴 -> policy evaluation 중단 기준 : value function의 변화량 < theta
            # theta가 클수록 필요한 episode 수는 작아지지만 iteration 수는 커짐
            # theta가 작을수록 필요한 episode 수는 커지지만 iteration 수는 작아짐
            # theta가 충분히 큰 수 일때는 episode, iteration 수가 일정해짐
        # target reward가 클수록 gamma를 작게 해줘야 함
            # gamma를 고정하고 target reward를 크게 하면 일정 크기에서 iteration이 끝나지 않음
        # iteration 최소값(최신) : 3
            # episode 수 고정 했을 때 보인 값
            # episode 수 고정 안했을 때 iteration 최소값(최신) : 9

    def translate_action_index(self, actions):
        pacman_action_index = []
        for i, action in enumerate(actions):
            pacman_action_index.append(i)
        return pacman_action_index

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
        self.rewards[temp : temp + 4] = 5

    def policy_iteration(self):
        iteration = 0
        self.all_step = 0
        self.all_episode = 0
        self.policy_stable = False
        while not self.policy_stable:
            self.policy_evaluation()
            self.policy_improvement()
            iteration += 1
            print('iteration : {}\n'.format(iteration))
        print('all_step : {}\n'.format(self.all_step))
        print('all_episode : {}\n'.format(self.all_episode))
        print('iteration : {}\n'.format(iteration))
        print('greedy_policy_matrix : \n{}\n'.format(self.greedy_policy_matrix))
        print('greedy_qvalue_matrix : \n{}\n'.format(self.greedy_qvalue_matrix.reshape((-1,4))))
        return self.greedy_policy_matrix

    def policy_evaluation(self):
        episode = 0
        while True:
            delta_qvalue = 0
            step = 0
            for now_index in range(self.state_num):
                state_present = self.transform_state_index(now_index)
                if state_present[0:2] == list(self.target_position):
                    qvalue_past = 0
                    qvalue_present = 0
                else:
                    qvalue_past = self.value_matrix[now_index]
                    qvalue_present = self.qvalue_update(now_index, state_present)
                    # print('qvalue_past : {}\n'.format(qvalue_past))
                    # print('qvalue_present : {}\n'.format(qvalue_present))
                delta_qvalue = max(delta_qvalue, abs(qvalue_past - qvalue_present))
                # print(abs(qvalue_past - qvalue_present))
                self.value_matrix[now_index] = qvalue_present
                #print('value_matrix : \n{}\n'.format(self.value_matrix.reshape((-1,4))))
                # print('delta_qvalue : {}\n'.format(delta_qvalue))
                step += 1
                #print('step : {}\n'.format(step))
            episode += 1
            self.all_step += step
            print('episode : {}\n'.format(episode))
            if delta_qvalue < self.theta: #episode == 10:
                self.all_episode += episode
                break
        self.greedy_qvalue_matrix = self.value_matrix

    def policy_improvement(self):
        self.policy_stable = True
        for now_index in range(self.state_num):
            state_present = self.transform_state_index(now_index)
            if state_present[0:2] != list(self.target_position):
                old_policy_matrix = copy.deepcopy(self.policy_matrix)
                greedy_action = self.policy_greedy_update(now_index, state_present)
                #print('policy_matrix : \n{}\n'.format(self.policy_matrix))
                if not np.array_equal(old_policy_matrix, self.policy_matrix):
                    self.policy_stable = False
        if self.policy_stable:
            self.greedy_policy_matrix = self.policy_matrix


    def qvalue_update(self, now_index, state_present):
        sum = 0
        for action in self.actions:
            next_index = self.next_state(state_present, action)
            self.value_func = self.policy_matrix[now_index][action] * (self.rewards[next_index] + self.gamma * self.value_matrix[next_index])
            sum += self.value_func
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
        return greedy_action

    def transform_state_index(self, state_index):
        row = state_index // (4 * 5)
        col = state_index % (4 * 5) // 4
        direction = state_index % (4 * 5) % 4
        return [row, col, direction]

    def next_state(self, state_present, action):
        state_next = copy.deepcopy(state_present)
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