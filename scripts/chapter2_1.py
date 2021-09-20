import numpy as np

class MDP():
    def __init__(self, canvas_grid):
        self.grid_dim = canvas_grid.grid_dim
        self.walls = canvas_grid.wall_lines()
        self.numState = 4 * self.grid_dim ** 2
        self.agent = canvas_grid.pacman
        # canvas_grid.pacman.position
        # canvas_grid.pacman.cardinal_point
        self.transitionProabilityMatrix()

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

        print(P)
        self.P = P


if __name__ == '__main__':
    print('input of MDP as canvas grid')