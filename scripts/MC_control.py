from scripts.pacman_entity import *


class MC_control():
    def __init__(self, pacman, numEpisode):
        self.epsilon = 1.0
        # self.learning_rate = 0.01
        self.gamma = 0.9
        self.memory = []
        self.numEpisode = numEpisode
        self.totReward = np.array([])

        self.pacman = pacman
        self.grid_dim = pacman.n
        self.numState = 4 * self.grid_dim ** 2
        self.action_list = [0, 1, 2]  # ["straight", "left", "right"]
        self.q_table = np.zeros((self.numState, len(self.action_list)))
        self.q_cnt = np.zeros((self.numState, len(self.action_list)))
        self.policy = np.zeros((self.numState, len(self.action_list)))

        self.policy_evaluation(self.numEpisode)

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

    def update(self):
        G_t = 0

        # V_t_old = self.value_table
        # V_t_new = self.value_table
        # for sample in reversed(self.memory):
        #     state = sample[0]
        #     reward = sample[1]
        #     G_t = reward + self.gamma * G_t
        #     V_t_new[state] = V_t_old[state] + self.learning_rate * (G_t - V_t_old[state])
        # self.value_table = V_t_new

        for sample in reversed(self.memory):
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            self.q_cnt[state][action] += 1

            G_t = reward + self.gamma * G_t
            Q_t = self.q_table[state][action]
            self.q_table[state][action] = Q_t + (G_t - Q_t) / self.q_cnt[state][action]

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
                    self.policy_improvement(episode)
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

        return self.policy

    def policy_improvement(self, episode):
        self.epsilon = 1.0 / (episode+1)

        for state in range(self.numState):
            max_value = np.amax(self.q_table[state])
            tie_Qchecker = np.where(self.q_table[state] == max_value)[0]

            if len(tie_Qchecker) > 1:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state,tie_Qchecker] = (1 - self.epsilon) / len(tie_Qchecker) + self.epsilon / len(self.action_list)
            else:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state,tie_Qchecker] = 1 - self.epsilon + self.epsilon / len(self.action_list)


if __name__ == "__main__":
    pacman = Pacman(5)
    MonteCarlo_policy = MC_control(pacman, 1000)
    # print(MonteCarlo_policy.policy_improvement())
    print(MonteCarlo_policy.q_table.reshape(-1, 3))
