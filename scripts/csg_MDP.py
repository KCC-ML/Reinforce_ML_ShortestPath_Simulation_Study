import numpy as np
from packman_control import World

class MDP:
    def __init__(self, canvas_grid):
        self.grid_dim = canvas_grid.grid_dim
        self.State_no = (self.grid_dim ** 2) * 4
        self.agent = canvas_grid.pacman
        self.walls = self.agent.wall_lines()
        self.reward = -1
        self.gamma = 0.9
        self.action = [0, 1, 2]
        self.action_probability = [0.8, 0.1, 0.1]

    # https://www.youtube.com/watch?v=3vuw23_l_oA&list=PLKs7xpqpX1beJ5-EOFDXTVckBQFFyTxUH&index=6 28:29
    # 위 링크 강의 영상에서 아래식을 가지고 코딩
    # vk+1 을 구하기 위해서 필요한 것 1)R파이 2)감마 3)P파이 4)vk

    # P파이 를 구하기 위한 메소드(P = 100x100의 s -> s'를 표현한 행렬)
    # policy_improvement 가 끝난 후에 다시 구해줘야 함
    def state_transition_matrix(self):
        P = np.zeros((self.State_no, self.State_no))
        for i in range(self.State_no)
            P[i]
            direction = i % 4
            row = (i // 4) // self.grid_dim
            col = (i // 4) % self.grid_dim

            if self.walls[direction, row, col] == 1:
                P[i, i - direction + ((direction - 1) % 4)] = self.action_probability[1]
                P[i, i - direction + ((direction + 1) % 4)] = self.action_probability[2]
                P[i, i] = self.action_probability[0]

            elif self.walls[direction, row, col] == 0:
                P[i, i - direction + ((direction - 1) % 4)] = self.action_probability[1]
                P[i, i - direction + ((direction + 1) % 4)] = self.action_probability[2]
                if direction % 4 == 0:
                    tmp = -4 * self.grid_dim
                elif direction % 4 == 1:
                    tmp = 4 * 1
                elif direction % 4 == 2:
                    tmp = 4 * self.grid_dim
                elif direction % 4 == 3:
                    tmp = -4 * 1
                P[i, i + tmp] = self.action_probability[0]

        if not np.all((P.sum(axis=1)) == 1):
            print("check transition probability matrix!!")

        # print("transition probability matrix: \n", P)
        self.P = P

    # vk(=v_post)를 구하기 위한 메소드(vk는 각 v(s)를 100x1의 벡터로 표현한 행렬)
    # vk+1(=v_next)을 구하기 위한 메소드(vk에 P를 곱하고, 감마를 곱하고
    # , R파이(R파이(=R) 는 R(=-1)을 100x1의 벡터로 표현한 행렬)를 더한다)
    def policy_evaluation(self):
        V_present = np.zeros((self.State_no, 1))
        R = np.ones((self.State_no, 1)) * -1
        V_next = R + self.gamma * self.P * V_present

        self.V_present = V_present
        self.V_next = V_next

    # 위에 구해진 vk+1로 agent 가 각 v(s)를 비교하여 가장 높은 v(s)를 택하는 action 을 취할 수 있도록 policy(0.8, 0.1, 0.1)를
    # improve 한다
    def policy_improvement(self):
        pass

    # 위 과정을 반복
    def policy_iteration(self):
        pass
