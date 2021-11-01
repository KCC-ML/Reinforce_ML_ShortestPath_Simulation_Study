import numpy as np
import gym
from collections import deque
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

class linearVFA_MCprediction():
    def __init__(self, iteration):
        self.env = gym.make('MountainCar-v0')
        self.que = deque([])

        self.learning_rate = 0.01
        self.gamma = 0.9
        self.iteration = iteration

        if os.path.isfile(os.getcwd() + '\\linearVFA_MCprediction_'+str(self.iteration)+'.npy'):
            self.linear_model = np.load('linearVFA_MCprediction_'+str(self.iteration)+'.npy')
        else:
            self.linear_model = np.zeros(self.env.observation_space.shape)
            self.policy_evaluation(self.iteration)
            np.save('linearVFA_MCprediction_'+str(self.iteration)+'.npy', self.linear_model)

    def update(self):
        self.que.reverse()

        G_t = 0
        for sample in self.que:
            state = sample[0]
            reward = sample[1]
            G_t = reward + self.gamma * G_t
            V_t = self.linearState(state)

            self.linear_model += self.learning_rate * (G_t - V_t) * state

        self.que.clear()

    def linearState(self, state):
        # print(state, self.linear_model)
        # print(np.dot(self.linear_model, state))
        return np.dot(self.linear_model, state)

    def policy_evaluation(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while True:
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                self.que.append([state, reward, done])

                state = next_state

                if done:
                    self.update()
                    break

            if episode % 10 == 0:
                print("{} episode done!".format(episode))

        return self.linear_model

    def policy_improvement(self, episode):
        self.epsilon = 1.0 / (episode + 1)

        for state in range(self.numState):
            max_value = np.amax(self.q_table[state])
            tie_Qchecker = np.where(self.q_table[state] == max_value)[0]

            if len(tie_Qchecker) > 1:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state, tie_Qchecker] = (1 - self.epsilon) / len(tie_Qchecker) + self.epsilon / len(
                    self.action_list)
            else:
                self.policy[state] = self.epsilon / len(self.action_list)
                self.policy[state, tie_Qchecker] = 1 - self.epsilon + self.epsilon / len(self.action_list)

    def draw_value_function(self):
        model = self.linear_model
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        x = np.linspace(-1.2, 0.6, 30) # pos
        y = np.linspace(-0.07, 0.07, 30) # vel

        X, Y = np.meshgrid(x, y)
        Z = model[0]*X + model[1]*Y

        ax.set_xlabel('pos')
        ax.set_xlim([-1.2, 0.6])

        ax.set_ylabel('vel')
        ax.set_ylim([-0.07, 0.07])

        ax.set_zlabel('value')

        # ax.contour(X, Y, Z, 50, cmap='binary')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        plt.show()


if __name__ == "__main__":
    iteration = 1000
    MCprediction = linearVFA_MCprediction(iteration)
    if os.path.isfile(os.getcwd()+'\\linearVFA_MCprediction_'+str(iteration)+'.npy'):
        coefficient = np.load('linearVFA_MCprediction_'+str(iteration)+'.npy')
    else:
        coefficient = MCprediction.policy_evaluation(iteration)
    print(coefficient)

    MCprediction.draw_value_function()

    env = gym.make('MountainCar-v0')
    env.reset()
    done = False
    max_position = -1.2
    while not done:
        next_values = np.array([])
        for action in range(3):
            next_state, reward, done, info = env.step(action)
            next_values = np.append(next_values, MCprediction.linearState(next_state))

        max_value = np.amax(next_values)
        tie_Qchecker = np.where(next_values == max_value)[0]

        if len(tie_Qchecker) > 1:
            idx = np.random.choice(tie_Qchecker, 1)[0]
        else:
            idx = np.argmax(next_values)

        action = idx

        res = env.step(action)
        if res[0][0] > max_position:
            max_position = res[0][0]
        env.render()
        print(max_position)

    env.close()
