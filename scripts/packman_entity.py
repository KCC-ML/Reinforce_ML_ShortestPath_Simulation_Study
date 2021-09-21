import numpy as np
import random

# set seed number to check env. goes right
# random.seed(13)
# np.random.seed(13)

class Env:
    def __init__(self, n):
        self.n = n
        self.percentage = 0.1
        self.gridmap = 0
        self.gridmap_goal = np.zeros([0,0])
        self._grid()
        self.generate_wall()

    def _grid(self):
        # create gridmap
        n = self.n
        gridmap = np.zeros((n, n))
        self.gridmap = gridmap
        self._goal()

    def _goal(self):
        # model-based
        self.gridmap[3][2] = 2
        self.gridmap_goal = np.array([3, 2])

        # model-free
        # n = self.n
        #
        # while True:
        #     e_x = random.randrange(n)
        #     e_y = random.randrange(n)
        #     if self.gridmap[e_y][e_x] == 0:
        #         self.gridmap[e_y][e_x] = 2
        #         self.gridmap_goal = np.array([e_y, e_x])
        #         break

    def goal_position(self):
        return self.gridmap_goal

    def generate_wall(self):
        # (direction, grid_row, grid_col), initiate with all zeros (no inner walls)
        self.walls = np.zeros((4, self.n, self.n))
        # change outer walls value as one
        self.walls[0, 0, :] = 1 # upper walls
        self.walls[1, :, self.n-1] = 1 # right walls
        self.walls[2, self.n-1, :] = 1 # lower walls
        self.walls[3, :, 0] = 1 # left walls

        # model-based
        wall_row = 3
        self.walls[0, wall_row, 0] = 1
        self.walls[2, wall_row, 1] = 1
        self.walls[3, wall_row, 2] = 1
        self.walls[0, wall_row, 2] = 1

        self.walls[2, wall_row - 1, 0] = 1
        self.walls[0, wall_row + 1, 1] = 1
        self.walls[1, wall_row, 1] = 1
        self.walls[2, wall_row - 1, 2] = 1

        # think
        # wall_row = 2
        # self.walls[1, wall_row, 2] = 1
        # self.walls[3, wall_row, 3] = 1
        # self.walls[0, wall_row, 3] = 1
        # self.walls[2, wall_row-1, 3] = 1
        #
        # wall_row = 1
        # self.walls[1, wall_row, 3] = 1
        # self.walls[3, wall_row, 4] = 1
        #
        # wall_row = 0
        # self.walls[1, wall_row, 2] = 1
        # self.walls[3, wall_row, 3] = 1

        # model-free
        # ratio = 0.1
        # tot_wall_num = 2 * self.grid_dim * (self.grid_dim - 1)
        # cnt = 0
        # while cnt < int(tot_wall_num * ratio):
        #     rand_num = random.randint(0, tot_wall_num-1)
        #     wall_direction = (rand_num // self.grid_dim**2)
        #     wall_row = ((rand_num % self.grid_dim**2) // self.grid_dim) - 1
        #     wall_col = ((rand_num % self.grid_dim**2) % self.grid_dim) - 1
        #
        #     if self.walls[wall_direction, wall_row, wall_col] == 1:
        #         continue
        #     elif self.walls[:, wall_row, wall_col].sum() == 3:
        #         continue
        #
        #     self.walls[wall_direction, wall_row, wall_col] = 1 # if zero there is no wall, if one there is a wall.
        #     if wall_direction == 0:
        #         self.walls[2, wall_row-1, wall_col] = 1
        #     elif wall_direction == 1:
        #         self.walls[3, wall_row, wall_col+1] = 1
        #     elif wall_direction == 2:
        #         self.walls[0, wall_row+1, wall_col] = 1
        #     elif wall_direction == 3:
        #         self.walls[1, wall_row, wall_col-1] = 1
        #     cnt += 1

    def wall_lines(self):
        return self.walls

class Pacman(Env):
    def __init__(self, n):
        super().__init__(n)
        self.cardinal_point = 0
        self.position = np.array([0,0])
        self.set_packman()

    # packman's movement
    def straight(self):
        cardinal_point = self.cardinal_point
        p_y = self.position[0]
        p_x = self.position[1]

        if cardinal_point == 1:
            if self.walls[1, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_x = p_x + 1
                tmp_p_y = p_y
        elif cardinal_point == 3:
            if self.walls[3, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_x = p_x - 1
                tmp_p_y = p_y
        elif cardinal_point == 2:
            if self.walls[2, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_y = p_y + 1
                tmp_p_x = p_x
        elif cardinal_point == 0:
            if self.walls[0, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_y = p_y - 1
                tmp_p_x = p_x

        if self.gridmap[tmp_p_y][tmp_p_x] == 0:
            self.gridmap[p_y][p_x] = 0
            self.position = np.array([tmp_p_y, tmp_p_x])
        elif self.gridmap[tmp_p_y][tmp_p_x] == 2:
            self.gridmap[p_y][p_x] = 0
            self.position = np.array([tmp_p_y, tmp_p_x])
            print('pacman arrives goal!!')
            return 1

    def left(self):
        print("pacman turns left")
        if self.cardinal_point == 1:
            self.cardinal_point = 0
        elif self.cardinal_point == 3:
            self.cardinal_point = 2
        elif self.cardinal_point == 2:
            self.cardinal_point = 1
        elif self.cardinal_point == 0:
            self.cardinal_point = 3
        return self.cardinal_point

    def right(self):
        print("pacman turns right")
        if self.cardinal_point == 1:
            self.cardinal_point = 2
        elif self.cardinal_point == 3:
            self.cardinal_point = 0
        elif self.cardinal_point == 2:
            self.cardinal_point = 3
        elif self.cardinal_point == 0:
            self.cardinal_point = 1
        return self.cardinal_point

    def set_packman(self):
        self.gridmap[0][0] = -1
        self.position = np.array([0, 0])
        self.cardinal_point = 0

        # model-free
        # n = self.n
        # gridmap = self.gridmap
        # cardinal_point_list = ["east", "west", "south", "north"]
        #
        # while True:
        #     p_x = random.randrange(0, n)
        #     p_y = random.randrange(0, n)
        #
        #     if gridmap[p_y][p_x] == 0:
        #         self.gridmap[p_y][p_x] = -1
        #         self.position = np.array([p_y, p_x])
        #         self.cardinal_point = cardinal_point_list[random.randrange(0, 4)]
        #         break

    def visualization(self):
        gridmap = self.gridmap.tolist()
        gridmap[self.gridmap_goal[0]][self.gridmap_goal[1]] = '*'

        if self.cardinal_point == 0:
            gridmap[self.position[0]][self.position[1]] = '^'
        elif self.cardinal_point == 1:
            gridmap[self.position[0]][self.position[1]] = '>'
        elif self.cardinal_point == 2:
            gridmap[self.position[0]][self.position[1]] = 'v'
        elif self.cardinal_point == 3:
            gridmap[self.position[0]][self.position[1]] = '<'

        for tmp in gridmap:
            print(tmp)

