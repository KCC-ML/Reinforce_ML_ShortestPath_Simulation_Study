import numpy as np


class MDP():
    def __init__(self, canvas_grid):
        self.grid_dim = canvas_grid.grid_dim
        self.numState = 4 * self.grid_dim ** 2
        self.agent = canvas_grid.pacman
        self.walls = self.agent.wall_lines()
        self.transitionProabilityMatrix()
        self.reward = -1
        self.gamma = 1
        self.action = [0, 1, 2]  # ["straight", "left", "right"]
        self.intialize_policy()

    def transitionProabilityMatrix(self):
        # initiate 100by100 matrix (|S|=100)
        P = np.zeros((self.numState, self.numState))

        # calculate transition probability for all states
        for i in range(self.numState):
            direction = i % 4
            row = (i // 4) // self.grid_dim
            col = (i // 4) % self.grid_dim

            # a wall exists in front of the agent
            if self.walls[direction, row, col] == 1:
                # set probability as 0.1 which states are turning left & turing right
                P[i, i - direction + (direction - 1) % 4] = 0.1
                P[i, i - direction + (direction + 1) % 4] = 0.1
                # set probability as 0.8 which state is not changed because agent meets the wall
                P[i, i] = 0.8

            # a wall doesn't exist in front of the agent
            elif self.walls[direction, row, col] == 0:
                # set probability as 0.1 which states are turning left & turing right
                P[i, i - direction + (direction - 1) % 4] = 0.1
                P[i, i - direction + (direction + 1) % 4] = 0.1
                # set probability as 0.8 which state is agent's heading direction
                if direction % 4 == 0:
                    tmp = -4 * self.grid_dim
                elif direction % 4 == 1:
                    tmp = 4 * 1
                elif direction % 4 == 2:
                    tmp = 4 * self.grid_dim
                elif direction % 4 == 3:
                    tmp = -4 * 1
                P[i, i + tmp] = 0.8

        # for checking whether sum of tpm is one
        if not np.all((P.sum(axis=1))==1):
            print("check transition probability matrix!!")

        # print("transition probability matrix: \n", P)
        self.P = P

    def intialize_policy(self):
        self.policy = np.empty([self.numState, len(self.action)], dtype=float)
        for i in range(self.policy.shape[0]):
            for j in range(self.policy.shape[1]):
                direction = i % 4
                row = (i // 4) // self.grid_dim
                col = (i // 4) % self.grid_dim
                if np.all(self.agent.gridmap_goal == np.array([row, col])):
                    self.policy[i][j] = 0.0
                else:
                    if j == 0:
                        self.policy[i][j] = 0.8
                    else:
                        self.policy[i][j] = 0.1
        # print("policy:\n",self.policy)

    def next_state(self, i, action):
        direction = i % 4
        row = (i // 4) // self.grid_dim
        col = (i // 4) % self.grid_dim

        if action == 0:
            if self.walls[direction, row, col] == 1:
                tmp = 0
            elif self.walls[direction, row, col] == 0:
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

        return i + tmp

    def policy_evaluation(self, iter_num):
        # initialize value_function
        post_value_table = np.zeros(self.numState)

        if iter_num == 0:
            print('Iteration: {} \n{}\n'.format(iter_num, post_value_table))
            return post_value_table

        for iteration in range(iter_num):
            next_value_table = np.zeros(self.numState, dtype=float)
            for i in range(self.numState):
                direction = i % 4
                row = (i // 4) // self.grid_dim
                col = (i // 4) % self.grid_dim
                if np.all(self.agent.gridmap_goal == np.array([row,col])):
                    value_t = 0
                else:
                    value_t = 0
                    for act in self.action:
                        tmp = self.next_state(i, act)
                        value = self.policy[i][act] * (self.reward + self.gamma * post_value_table[tmp])

                        value_t += value

                next_value_table[i] = round(value_t, 3)

            post_value_table = next_value_table
            iteration += 1

            tmp = next_value_table.reshape((-1,4))
            if (iteration % 10) != iter_num:
                if iteration > 100:
                    if (iteration % 20) == 0:
                        print('Iteration: {} \n{}\n'.format(iteration, tmp))
                else:
                    if (iteration % 10) == 0:
                        print('Iteration: {} \n{}\n'.format(iteration, tmp))

        return next_value_table

    def policy_improvement(self, value):
        action_match = ['straight', 'left', 'right']
        action_table = []
        policy = self.policy

        for i in range(self.numState):
            # init state_action value function
            q_func_list = []
            direction = i % 4
            row = (i // 4) // self.grid_dim
            col = (i // 4) % self.grid_dim

            if np.all(self.agent.gridmap_goal == np.array([row, col])):
                action_table.append('T')
            else:
                for act in self.action:
                    # tmp is number of next state
                    tmp = self.next_state(i, act)
                    # append state_action value
                    q_func_list.append(value[tmp])

                # take action which has maximum state_value function in the next state
                # if there are more than one action with same state_action value, take all of them
                max_actions = [action_v for action_v, x in enumerate(q_func_list) if x == max(q_func_list)]

                # update policy
                policy[i] = [0] * len(self.action)
                for y in max_actions:
                    policy[i][y] = 1 / len(max_actions)

                # get action
                idx = np.argmax(policy[i])
                action_table.append(action_match[idx])

        tmp = np.asarray(action_table).reshape((-1,4))

        print('Updated policy is :\n{}\n'.format(policy))
        print('at each state, chosen action is :\n{}'.format(tmp))

        return policy

if __name__ == '__main__':
    print('input of MDP as canvas grid')