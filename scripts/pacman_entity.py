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

class Pacman(Env):
    def __init__(self, n):
        super().__init__(n)
        self.cardinal_point = "north"
        self.position = np.array([0,0])
        self.set_packman()

    # packman's movement
    def straight(self, wall_lines, agent_coordinate):
        cardinal_point = self.cardinal_point
        p_y = self.position[0]
        p_x = self.position[1]

        if cardinal_point == "east":
            if wall_lines[1, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_x = p_x + 1
                tmp_p_y = p_y
        elif cardinal_point == "west":
            if wall_lines[3, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_x = p_x - 1
                tmp_p_y = p_y
        elif cardinal_point == "south":
            if wall_lines[2, p_y, p_x] == 1:
                print("pacman goes forward but meets wall")
                tmp_p_x = p_x
                tmp_p_y = p_y
            else:
                print('pacman goes forward')
                tmp_p_y = p_y + 1
                tmp_p_x = p_x
        elif cardinal_point == "north":
            if wall_lines[0, p_y, p_x] == 1:
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
        if self.cardinal_point == "east":
            self.cardinal_point = "north"
        elif self.cardinal_point == "west":
            self.cardinal_point = "south"
        elif self.cardinal_point == "south":
            self.cardinal_point = "east"
        elif self.cardinal_point == "north":
            self.cardinal_point = "west"
        return self.cardinal_point

    def right(self):
        print("pacman turns right")
        if self.cardinal_point == "east":
            self.cardinal_point = "south"
        elif self.cardinal_point == "west":
            self.cardinal_point = "north"
        elif self.cardinal_point == "south":
            self.cardinal_point = "west"
        elif self.cardinal_point == "north":
            self.cardinal_point = "east"
        return self.cardinal_point

    def set_packman(self):
        self.gridmap[0][0] = -1
        self.position = np.array([0, 0])
        self.cardinal_point = "north"

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

        if self.cardinal_point == 'north':
            gridmap[self.position[0]][self.position[1]] = '^'
        elif self.cardinal_point == 'east':
            gridmap[self.position[0]][self.position[1]] = '>'
        elif self.cardinal_point == 'south':
            gridmap[self.position[0]][self.position[1]] = 'v'
        elif self.cardinal_point == 'west':
            gridmap[self.position[0]][self.position[1]] = '<'

        for tmp in gridmap:
            print(tmp)

