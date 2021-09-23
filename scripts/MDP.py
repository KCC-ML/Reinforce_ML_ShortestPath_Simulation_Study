class MDP:
    def __init__(self, state_num, actions):
        # 불변값 : 사용자 지정값, 환경값 정의
        self.state_num
        self.grid_dim
        self.actions
        self.start_policy = [1.0, 0.8, 0.8]
        self.rewards
        self.target_position # (3,2)
        self.gamma = 1
        self.theta = 0.001
        self.initialize_elements()

    def initialize_elements(self):
        self.matrixization_state()
        self.matrixization_value()
        self.matrixization_policy()
        self.matrixization_reward()

    def matrixiation_state(self):
        self.state_num

    def matrixization_value(self):
        self.value_matrix = np.zeros((self.state_num, 1))

    def matrixization_policy(self):
        self.policy_matrix = np.reshape(self.start_policy * self.state_num, (self.state_num, self.start_policy))
        temp = 4 * 5 * self.target_position[0] + 4 * self.target_position[1]
        self.policy_matrix[temp : temp + 4, :] = 0

    def matrixization_reward(self):
        self.rewards = -1 * np.ones((self.state_num, 1))
        temp = 4 * 5 * self.target_position[0] + 4 * self.target_position[1]
        self.rewards[temp : temp + 4] = 0

    def policy_iteration(self):
        self.policy_stable = False
        while not self.policy_stable:
            self.policy_evaluation()
            self.policy_improvement()
        return self.greedy_policy_matrix

    def policy_evaluation(self):
        while delta_qvalue < self.theta:
            delta_qvalue = 0
            for state in :
                qvalue_past = qvalue_present
                qvalue_present = self.qvalue_update()
                delta_qvalue = max(delta_qvalue, abs(qvalue_past - qvalue_present))
        self.greedy_qvalue_matrix = qvalue_matrix

    def policy_improvement(self):
        for (allstate):
            old_action = policy_matrix[now]
            policy_matrix[now] = np.argmax()
            if old_action == policy_matrix[now]:
                self.policy_stable = True
                self.greedy_policy_matrix = policy_matrix

    def qvalue_update(self):
        sum = 0
        for action in self.greedy_policy_matrix[now]:
            sum += self.reward[now] + self.gamma * action * value_matrix[now]
        qvalue_present = sum
        return qvalue_present

