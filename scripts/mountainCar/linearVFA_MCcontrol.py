import numpy as np
import gym
from collections import deque
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

class linearVFA_MCcontrol():
    def __init__(self, iteration):
        self.env = gym.make('MountainCar-v0')
        self.que = deque([])

        self.learning_rate = 0.01
        self.gamma = 0.9
        self.iteration = iteration

        if os.path.isfile(os.getcwd() + '\\linearVFA_MCcontrol_' + str(self.iteration) + '.npy'):
            self.linear_model = np.load('linearVFA_MCcontrol_' + str(self.iteration) + '.npy')
        else:
            self.linear_model = np.zeros(self.env.observation_space.shape[0]+1)
            self.policy_evaluation(self.iteration)
            np.save('linearVFA_MCcontrol_' + str(self.iteration) + '.npy', self.linear_model)

    def update(self):
        self.que.reverse()

        G_t = 0
        for sample in self.que:
            q_value = np.append(sample[0], sample[1]) # [state, action]
            reward = sample[2]
            G_t = reward + self.gamma * G_t
            q_t = self.linearState(q_value)

            self.linear_model += self.learning_rate * (G_t - q_t) * q_value

        self.que.clear()

    def linearState(self, q_value):
        # print(state, self.linear_model)
        # print(np.dot(self.linear_model, state))
        return np.dot(self.linear_model, q_value)

    def policy_evaluation(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False

            while True:
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                self.que.append([state, action, reward, done])

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

    def draw_value_function(self, model):
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
    iteration = 10000
    MCcontrol = linearVFA_MCcontrol(iteration)
    if os.path.isfile(os.getcwd()+'\\linearVFA_MCcontrol_'+str(iteration)+'.npy'):
        coefficient = np.load('linearVFA_MCcontrol_'+str(iteration)+'.npy')
    else:
        coefficient = MCcontrol.policy_evaluation(iteration)
    print(coefficient)

    # linearVFA_MCcontrol().draw_value_function(coefficient)

    env = gym.make('MountainCar-v0')
    state = env.reset()
    done = False
    max_position = -1.2
    while not done:
        next_q_values = np.array([])
        for action in range(3):
            q_value = state
            feature_vector = np.append(q_value, action)
            next_q_values = np.append(next_q_values, MCcontrol.linearState(feature_vector))

        max_value = np.amax(next_q_values)
        tie_Qchecker = np.where(next_q_values == max_value)[0]

        if len(tie_Qchecker) > 1:
            idx = np.random.choice(tie_Qchecker, 1)[0]
        else:
            idx = np.argmax(next_q_values)

        action = idx

        res = env.step(action)
        if res[0][0] > max_position:
            max_position = res[0][0]
        env.render()
        print(max_position)

    env.close()
