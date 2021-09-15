import numpy as np
import random
import time
from simulation_entity import CanvasGrid
# set seed number to check env. goes right
# random.seed(13)
# np.random.seed(13)


class Env():
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
        # self._wall()
        self._goal()

    def _wall(self):
        # The total number of wall should be less than 'percentage'.
        # However, there should not be isolated grid which means there is no closed grid.
        percentage = self.percentage
        n = self.n
        gridmap = self.gridmap

        wall_sum = int((self.n**2) * percentage)
        cnt_wall = 0
        while cnt_wall <= wall_sum:
            w_x = random.randrange(1, n)
            w_y = random.randrange(1, n)

            # Check the grid is closed.
            if 2 <= w_x <= n - 3 and 2 <= w_y <= n - 3:
                if int(gridmap[w_y + 1][w_x - 1]) + int(gridmap[w_y + 1][w_x + 1]) + int(gridmap[w_y + 2][w_x]) == 3 \
                        or int(gridmap[w_y - 1][w_x - 1]) + int(gridmap[w_y + 1][w_x - 1]) + int(
                    gridmap[w_y][w_x - 2]) == 3 \
                        or int(gridmap[w_y - 1][w_x - 1]) + int(gridmap[w_y - 1][w_x + 1]) + int(
                    gridmap[w_y - 2][w_x]) == 3 \
                        or int(gridmap[w_y - 1][w_x + 1]) + int(gridmap[w_y + 1][w_x + 1]) + int(
                    gridmap[w_y][w_x + 2]) == 3:
                    continue
                else:
                    gridmap[w_y][w_x] = 1
                    cnt_wall += 1
            else:
                gridmap[w_y][w_x] = 1
                cnt_wall += 1
        self.gridmap = gridmap
        self._goal()

    def _goal(self):
        n = self.n

        while True:
            e_x = random.randrange(n)
            e_y = random.randrange(n)
            if self.gridmap[e_y][e_x] == 0:
                self.gridmap[e_y][e_x] = 2
                self.gridmap_goal = np.array([e_y, e_x])
                break
        print(self.gridmap_goal)

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
            tmp_p_x = p_x + 1
            tmp_p_y = p_y
        elif cardinal_point == "west":
            tmp_p_x = p_x - 1
            tmp_p_y = p_y
        elif cardinal_point == "south":
            tmp_p_y = p_y + 1
            tmp_p_x = p_x
        elif cardinal_point == "north":
            tmp_p_y = p_y - 1
            tmp_p_x = p_x

        print(wall_lines[0]['start_x'], wall_lines[0]['end_x'])
        print(wall_lines)
        line_len = wall_lines[0]['length']
        for wall in wall_lines:
            print(line_len, -line_len)
            if cardinal_point == "east" and wall['dirt'] == 'height':
                if (line_len > (wall['start_x'] - agent_coordinate[0]) > 0) and (wall['start_y'] < agent_coordinate[1] < wall['end_y']):
                    print("pacman goes forward but meets wall")
                    return None
            elif cardinal_point == "west" and wall['dirt'] == 'height' and wall['start_x'] < agent_coordinate[0]:
                if (0 > (wall['start_x'] - agent_coordinate[0]) > (-line_len)) and (wall['start_y'] < agent_coordinate[1] < wall['end_y']):
                    print("pacman goes forward but meets wall")
                    return None
            elif cardinal_point == "south" and wall['dirt'] == 'width' and wall['start_y'] > agent_coordinate[1]:
                if (line_len > (wall['start_y'] - agent_coordinate[1]) > 0) and (wall['start_x'] < agent_coordinate[0] < wall['end_x']):
                    print("pacman goes forward but meets wall")
                    return None
            elif cardinal_point == "north" and wall['dirt'] == 'width' and wall['start_y'] < agent_coordinate[1]:
                if (0 > (wall['start_y'] - agent_coordinate[1]) > (-line_len)) and (wall['start_x'] < agent_coordinate[0] < wall['end_x']):
                    print("pacman goes forward but meets wall")
                    return None

        if tmp_p_x < 0 or tmp_p_x >= self.n or tmp_p_y < 0 or tmp_p_y >= self.n:
            print("pacman goes forward but meets wall")
        elif self.gridmap[tmp_p_y][tmp_p_x] == 0:
            self.gridmap[p_y][p_x] = 0
            self.position = np.array([tmp_p_y, tmp_p_x])
            print('pacman goes forward')
        elif self.gridmap[tmp_p_y][tmp_p_x] == 2:
            self.gridmap[p_y][p_x] = 0
            self.position = np.array([tmp_p_y, tmp_p_x])
            print('pacman arrives goal!!')
            return 1
        else:
            print("pacman goes forward but meets wall")

        # return p_x, p_y

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
        n = self.n
        gridmap = self.gridmap
        cardinal_point_list = ["east", "west", "south", "north"]

        while True:
            p_x = random.randrange(1, n)
            p_y = random.randrange(1, n)

            if gridmap[p_y][p_x] == 0:
                self.gridmap[p_y][p_x] = -1
                self.position = np.array([p_y, p_x])
                self.cardinal_point = cardinal_point_list[random.randrange(0, 4)]
                break

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

