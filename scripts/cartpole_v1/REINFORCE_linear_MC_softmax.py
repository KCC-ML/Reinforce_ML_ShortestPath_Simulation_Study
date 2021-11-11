import gym
import numpy as np
import matplotlib.pyplot as plt


class Policy():
    def __init__(self, env, learning_rate=0.001, gamma=0.9):
        super(Policy, self).__init__()
        self.data = []
        self.gamma = gamma
        self.learning_rate = learning_rate
        self._env = env

        self.weight = np.zeros((self._env.observation_space.shape[0], self._env.action_space.n))

    def policy(self, state):
        action = state.dot(self.weight)
        exp_action = np.exp(action)
        prob = exp_action / np.sum(exp_action)
        return prob

    def put_data(self, item):
        self.data.append(item)

    def update(self):
        G_t = 0
        for state, action, prob, reward in self.data[::-1]:
            G_t = reward + self.gamma * G_t

            dsoftmax = self.softmax_grad(prob)[action, :]

            dlog_softmax = dsoftmax / prob[action]
            state = state.reshape((-1, 1))
            dlog_softmax = dlog_softmax.reshape((1, -1))
            grad = state.dot(dlog_softmax)

            self.weight += self.learning_rate * grad * G_t

        self.data = []

    def softmax_grad(self, softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


if __name__=='__main__':
    env = gym.make('CartPole-v1')
    agent = Policy(env)
    rewards = []

    num_epi = 1000
    for i_epi in range(num_epi):
        state = env.reset()
        tot_reward = 0
        done = False

        while not done:
            prob = agent.policy(state)
            action = np.random.choice(2, p=prob)
            next_state, reward, done, _ = env.step(action)

            agent.data.append([state, action, prob, reward])
            tot_reward += reward
            # env.render()

            state = next_state

        rewards.append(tot_reward)
        agent.update()

        if (i_epi + 1) % 20 == 0:
            print(f"Policy update with {i_epi + 1} with reward {tot_reward}!")

    env.close()

    fig = plt.subplots()

    x = np.arange(num_epi)
    y = rewards

    plt.plot(x, y, color='blue')
    # plt.savefig('REINFORCE_linear_MC_softmax_result.png')
    plt.show()