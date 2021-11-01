import gym
import numpy as np

env = gym.make('MountainCar-v0')

class value_Estimator():
    def __init__(self, sample_state):
        self.state = sample_state
        self.weight = np.zeros(len(sample_state))
        self.v_hat = 0
        self.learning_rate = 0.01

    def value_(self, state):
        value = np.dot(state, self.weight)

        return value

    def delta_w(self):
        dw = self.learning_rate * (v_target - value) * state