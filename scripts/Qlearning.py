from scripts.pacman_entity import *
import matplotlib.pyplot as plt


class Qlearning():
    def __init__(self, pacman, numEpisode):
        self.epsilon = 1.0
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.totReward = np.array([])

        self.pacman = pacman
        self.grid_dim = pacman.n
        self.numState = 4 * self.grid_dim ** 2
        self.action_list = [0, 1, 2]  # ["straight", "left", "right"]

        self.q_table = np.zeros((self.numState, len(self.action_list)))
        self.policy = np.zeros((self.numState, len(self.action_list)))

        self.policy_evaluation(numEpisode)

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

    def update(self, state, action, reward, next_state, next_action):
        # G_t = reward + self.gamma * self.q_table[next_state][next_action]
        G_t = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (G_t - self.q_table[state][action])

    def policy_evaluation(self, num_episode):
        plt.ion()
        figure, ax = plt.subplots()
        rewards = []

        # using action value function
        for episode in range(1, num_episode+1):
            state = self.pacman.reset()
            action = self.get_action(state)
            done = False
            step = 0
            tot_reward = 0

            while True:
                next_state, reward, done = self.pacman.step(state, action)
                next_action = self.get_action(next_state)

                self.update(state, action, reward, next_state, next_action)
                self.policy_improvement(episode)

                tot_reward += reward

                step += 1
                state = next_state
                action = next_action

                if done:
                    rewards.append(tot_reward)
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

            x = np.arange(episode)
            y = rewards
            line1, = ax.plot(x, y, color='blue')
            figure.canvas.draw()
            figure.canvas.flush_events()

        return self.policy

    def policy_improvement(self, episode):
        self.epsilon = 1.0 / (episode+1)

        for state in range(self.numState):
            max_value = np.amax(self.q_table[state])
            tie_Qchecker = np.where(self.q_table[state] == max_value)[0]

            if len(tie_Qchecker) > 1:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state, tie_Qchecker] = (1 - self.epsilon) / len(tie_Qchecker) + self.epsilon / len(self.action_list)
            else:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state, tie_Qchecker] = 1 - self.epsilon + self.epsilon / len(self.action_list)


if __name__ == "__main__":
    pacman = Pacman(5)
    Qlearning = Qlearning(pacman, 1000)
    print(Qlearning.q_table.reshape(-1, 3))
