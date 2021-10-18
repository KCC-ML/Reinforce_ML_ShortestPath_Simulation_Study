from scripts.pacman_entity import *


class MC_control():
    def __init__(self, pacman, numEpisode):
        self.epsilon = 0.1
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.memory = []
        self.numEpisode = numEpisode
        self.totReward = np.array([])

        self.pacman = pacman
        self.grid_dim = pacman.n
        self.numState = 4 * self.grid_dim ** 2
        self.action_list = [0, 1, 2]  # ["straight", "left", "right"]
        self.q_table = np.zeros((self.numState, len(self.action_list)))
        self.policy = np.zeros((self.numState, len(self.action_list)))
        self.policy[0][:] = 0.8
        self.policy[1][:] = 0.1
        self.policy[2][:] = 0.1

        self.policy_evaluation(self.numEpisode)
        self.policy_improvement()

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

    def update(self):
        method = 'last_visit'  # ['first_visit', 'every_visit', 'last_visit']
        G_t = 0

        if method == 'first_visit':
            V_t_old = self.q_table
            V_t_new = self.q_table
            for sample in reversed(self.memory):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                G_t = reward + self.gamma * G_t
                V_t_new[state][action] = V_t_old[state][action] + self.learning_rate * (G_t - V_t_old[state][action])
            self.q_table = V_t_new
        elif method == 'every_visit':
            for sample in reversed(self.memory):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                G_t = reward + self.gamma * G_t
                V_t = self.q_table[state][action]
                self.q_table[state][action] = V_t + self.learning_rate * (G_t - V_t)
        elif method == 'last_visit':
            visit_states = []

            for sample in reversed(self.memory):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                G_t = reward + self.gamma * G_t
                V_t = self.q_table[state][action]
                if state not in visit_states:
                    visit_states.append(state)
                    self.q_table[state][action] = V_t + self.learning_rate * (G_t - V_t)

    def memorizer(self, state, action, reward, done):
        self.memory.append([state, action, reward, done])

    def policy_evaluation(self, num_episode):
        # using action value function
        for episode in range(num_episode):
            state = self.pacman.reset()
            done = False
            step = 0

            while True:
                action = self.get_action(state)
                next_state, reward, done = self.pacman.step(state, action)

                self.memorizer(state, action, reward, done)

                step += 1
                state = next_state

                if done:
                    self.update()
                    self.memory.clear()
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

    def policy_improvement(self):
        for state in range(self.numState):
            max_value = np.amax(self.q_table[state])
            tie_Qchecker = np.where(self.q_table[state] == max_value)[0]

            if len(tie_Qchecker) > 1:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state,tie_Qchecker] = 1 - self.epsilon + self.epsilon / len(self.action_list)
            else:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state,tie_Qchecker] = 1 - self.epsilon + self.epsilon / len(self.action_list)

        policy = self.policy
        return policy


if __name__ == "__main__":
    pacman = Pacman(5)
    MonteCarlo_policy = MC_control(pacman, 1000)
    # print(MonteCarlo_policy.policy_improvement())
    print(MonteCarlo_policy.q_table.reshape(-1, 3))
