import numpy as np
from packman_control_5by5 import World

class MDP:
    def __init__(self, canvas_grid):
        self.grid_dim = canvas_grid.grid_dim
        self.State_sum = (self.grid_dim ** 2) * 4
        self.agent = canvas_grid.pacman
        self.walls = canvas_grid.wall_lines()
        self.gamma = 0.9
        self.action = [0, 1, 2]
        self.theta = 0.00001
        self.target_reward = 0
        self.reward = np.ones((self.State_sum, 1)) * -1
        for i in range(69, 73):
            self.reward[i, 0] = self.target_reward
        self.initial_policy = np.zeros((100, 3))
        self.initial_P = np.zeros((self.State_sum, self.State_sum))
        self.initial_V = np.zeros((self.State_sum, 1))
        self.iteration = 1

    # https://www.youtube.com/watch?v=3vuw23_l_oA&list=PLKs7xpqpX1beJ5-EOFDXTVckBQFFyTxUH&index=6 28:29
    # 위 링크 강의 영상에서 아래식을 가지고 코딩
    # vk+1 을 구하기 위해서 필요한 것 1)R파이 2)감마 3)P파이 4)vk

    # P파이 를 구하기 위한 메소드(P = 100x100의 s -> s'를 표현한 행렬)
    # policy_improvement 가 끝난 후에 다시 구해줘야 함
    def state_transition_matrix(self, post_P, improved_policy):
        P = post_P
        for State_no in range(self.State_sum):
            direction = State_no % 4
            row = (State_no // 4) // self.grid_dim
            col = (State_no // 4) % self.grid_dim

            if self.walls[direction, row, col] == 1:
                P[State_no, State_no - direction + ((direction - 1) % 4)] = improved_policy[State_no, 1]
                P[State_no, State_no - direction + ((direction + 1) % 4)] = improved_policy[State_no, 2]
                P[State_no, State_no] = improved_policy[State_no, 0]

            elif self.walls[direction, row, col] == 0:
                P[State_no, State_no - direction + ((direction - 1) % 4)] = improved_policy[State_no, 1]
                P[State_no, State_no - direction + ((direction + 1) % 4)] = improved_policy[State_no, 2]
                if direction % 4 == 0:
                    tmp = -4 * self.grid_dim
                elif direction % 4 == 1:
                    tmp = 4 * 1
                elif direction % 4 == 2:
                    tmp = 4 * self.grid_dim
                elif direction % 4 == 3:
                    tmp = -4 * 1
                P[State_no, State_no + tmp] = improved_policy[State_no, 0]

        # if not np.all((P.sum(axis=1)) == 1):
        #     print("check transition probability matrix!!")

        # self.P = P
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print("transition probability matrix: \n", P)
        return P

    # vk(=v_post)를 구하기 위한 메소드(vk는 각 v(s)를 100x1의 벡터로 표현한 행렬)
    # vk+1(=v_next)을 구하기 위한 메소드(vk에 P를 곱하고, 감마를 곱하고
    # , R파이(R파이(=R) 는 R(=-1)을 100x1의 벡터로 표현한 행렬)를 더한다)
    def policy_evaluation(self, R, P, post_V):
        V = R + self.gamma * P @ post_V
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        return V

    # 위에 구해진 vk+1로 agent 가 각 v(s)를 비교하여 가장 높은 v(s)를 택하는 action 을 취할 수 있도록 policy(0.8, 0.1, 0.1)를
    # improve 한다
    def policy_improvement(self, post_policy, next_V, iteration):
        improved_policy = post_policy
        if iteration == 0:
            for i in range(100):
                for j in range(3):
                    if j == 0:
                        improved_policy[i, j] = 0.8
                    else:
                        improved_policy[i, j] = 0.1
            return improved_policy
        else:
            for State_no in range(100):
                direction = State_no % 4
                row = (State_no // 4) // self.grid_dim
                col = (State_no // 4) % self.grid_dim
                if self.walls[direction, row, col] == 1:
                    v_left = next_V[State_no - direction + ((direction - 1) % 4), 0]
                    v_right = next_V[State_no - direction + ((direction + 1) % 4), 0]
                    v_straight = next_V[State_no, 0]

                    if v_straight == v_left and v_straight == v_right:
                        improved_policy[State_no] = [1/3, 1/3, 1/3]
                    elif v_straight > v_left and v_straight > v_right:
                        improved_policy[State_no] = [1, 0, 0]
                    elif v_left > v_straight and v_left > v_right:
                        improved_policy[State_no] = [0, 1, 0]
                    elif v_right > v_straight and v_right > v_left:
                        improved_policy[State_no] = [0, 0, 1]
                    elif v_straight == v_left and v_left > v_right:
                        improved_policy[State_no] = [1/2, 1/2, 0]
                    elif v_straight == v_right and v_right > v_left:
                        improved_policy[State_no] = [1/2, 0, 1/2]
                    elif v_left == v_right and v_left > v_straight:
                        improved_policy[State_no] = [0, 1/2, 1/2]

                elif self.walls[direction, row, col] == 0:
                    v_left = next_V[State_no - direction + ((direction - 1) % 4), 0],
                    v_right = next_V[State_no - direction + ((direction + 1) % 4), 0],
                    if direction % 4 == 0:
                        tmp = -4 * self.grid_dim
                    elif direction % 4 == 1:
                        tmp = 4 * 1
                    elif direction % 4 == 2:
                        tmp = 4 * self.grid_dim
                    elif direction % 4 == 3:
                        tmp = -4 * 1
                    v_straight = next_V[State_no + tmp, 0]

                    if v_straight == v_left and v_straight == v_right:
                        improved_policy[State_no] = [1/3, 1/3, 1/3]
                    elif v_straight > v_left and v_straight > v_right:
                        improved_policy[State_no] = [1, 0, 0]
                    elif v_left > v_straight and v_left > v_right:
                        improved_policy[State_no] = [0, 1, 0]
                    elif v_right > v_straight and v_right > v_left:
                        improved_policy[State_no] = [0, 0, 1]
                    elif v_straight == v_left and v_left > v_right:
                        improved_policy[State_no] = [1/2, 1/2, 0]
                    elif v_straight == v_right and v_right > v_left:
                        improved_policy[State_no] = [1/2, 0, 1/2]
                    elif v_left == v_right and v_left > v_straight:
                        improved_policy[State_no] = [0, 1/2, 1/2]
            return improved_policy

    # 위 과정을 반복
    def policy_iteration(self):
        print("iteration: ", self.iteration)
        initial_policy = self.policy_improvement(self.initial_policy, self.initial_V, 0)
        next_P = self.state_transition_matrix(self.initial_P, initial_policy)
        next_V = self.policy_evaluation(self.reward, next_P, self.initial_V)
        improved_policy = self.policy_improvement(initial_policy, next_V, self.iteration)

        while True:
            self.iteration += 1
            print("iteration: ", self.iteration)
            post_V = next_V
            next_P = self.state_transition_matrix(next_P, improved_policy)
            next_V = self.policy_evaluation(self.reward, next_P, post_V)
            improved_policy = self.policy_improvement(improved_policy, next_V, self.iteration)

            if self.theta > abs(min(next_V - post_V)):
                optimal_policy = improved_policy
                optimal_V = next_V
                break

        print("optimal_policy:", optimal_policy)
        print("optimal_V:", optimal_V)

        return optimal_policy
