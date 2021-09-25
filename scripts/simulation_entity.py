from tkinter import *
import random
import PIL.Image, PIL.ImageTk
import numpy as np

class WindowTkinter:
    def __init__(self):
        self.window = Tk()

    def create_window(self):
        self.window.title("grid world")
        self.window.geometry("500x500")
        self.window.resizable(True, True)
        return self.window

class CanvasGrid:
    def __init__(self, window, agent):
        self.window = window
        self.pacman = agent
        self.grid_dim = self.pacman.n
        # self.window = Tk()
        self.run()

    def run(self):
        self.create_canvas()
        self.generate_wall()
        self.draw_line()

    def create_canvas(self):
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = Canvas(self.window, width = self.canvas_width, height=self.canvas_height, bg="black", bd=2)
        self.canvas.pack(fill = "both", expand = True)

    def draw_line(self):
        self.line_len = int((self.canvas_width - 40) / self.grid_dim)

        white_wall_index = np.where(self.walls == 0)
        for idx, _ in enumerate(white_wall_index[0]):
            if white_wall_index[0][idx] == 0:
                start_x = 20 + white_wall_index[2][idx] * self.line_len
                start_y = 20 + white_wall_index[1][idx] * self.line_len
                end_x = start_x + self.line_len
                end_y = start_y
            elif white_wall_index[0][idx] == 1:
                start_x = 20 + (white_wall_index[2][idx] + 1) * self.line_len
                start_y = 20 + white_wall_index[1][idx] * self.line_len
                end_x = start_x
                end_y = start_y + self.line_len
            elif white_wall_index[0][idx] == 2:
                start_x = 20 + (white_wall_index[2][idx] + 1) * self.line_len
                start_y = 20 + (white_wall_index[1][idx] + 1) * self.line_len
                end_x = start_x - self.line_len
                end_y = start_y
            elif white_wall_index[0][idx] == 3:
                start_x = 20 + white_wall_index[2][idx] * self.line_len
                start_y = 20 + (white_wall_index[1][idx] + 1) * self.line_len
                end_x = start_x
                end_y = start_y - self.line_len
            self.canvas.create_line(start_x, start_y, end_x, end_y, fill='white')

        wall_index = np.where(self.walls == 1)
        for idx, _ in enumerate(wall_index[0]):
            if wall_index[0][idx] == 0:
                start_x = 20 + wall_index[2][idx] * self.line_len
                start_y = 20 + wall_index[1][idx] * self.line_len
                end_x = start_x + self.line_len
                end_y = start_y
            elif wall_index[0][idx] == 1:
                start_x = 20 + (wall_index[2][idx] + 1) * self.line_len
                start_y = 20 + wall_index[1][idx] * self.line_len
                end_x = start_x
                end_y = start_y + self.line_len
            elif wall_index[0][idx] == 2:
                start_x = 20 + (wall_index[2][idx] + 1) * self.line_len
                start_y = 20 + (wall_index[1][idx] + 1) * self.line_len
                end_x = start_x - self.line_len
                end_y = start_y
            elif wall_index[0][idx] == 3:
                start_x = 20 + wall_index[2][idx] * self.line_len
                start_y = 20 + (wall_index[1][idx] + 1) * self.line_len
                end_x = start_x
                end_y = start_y - self.line_len

            self.canvas.create_line(start_x, start_y, end_x, end_y, fill='blue', width=5)

    def generate_wall(self):
        # (direction, grid_row, grid_col), initiate with all zeros (no inner walls)
        self.walls = np.zeros((4, self.grid_dim, self.grid_dim))
        # change outer walls value as one
        self.walls[0, 0, :] = 1 # upper walls
        self.walls[1, :, self.grid_dim-1] = 1 # right walls
        self.walls[2, self.grid_dim-1, :] = 1 # lower walls
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

    def set_agent(self, agent, cardinal_point):
        if cardinal_point == "east":
            img = PIL.Image.open("../figures/pngegg2.png")
        elif cardinal_point == "west":
            img = PIL.Image.open("../figures/pngegg2.png").transpose(PIL.Image.ROTATE_180)
        elif cardinal_point == "south":
            img = PIL.Image.open("../figures/pngegg2.png").transpose(PIL.Image.ROTATE_270)
        elif cardinal_point == "north":
            img = PIL.Image.open("../figures/pngegg2.png").transpose(PIL.Image.ROTATE_90)

        resized_image = img.resize((self.line_len-10, self.line_len-10))
        self.agent_image = PIL.ImageTk.PhotoImage(resized_image, master=self.window)

        self.agent_y = int(20 + self.line_len * agent.position[0] + self.line_len/2)
        self.agent_x = int(20 + self.line_len * agent.position[1] + self.line_len/2)
        self.canvas.create_image(self.agent_x, self.agent_y, anchor=CENTER, image=self.agent_image)

    def agent_coordinate(self):
        return [self.agent_x, self.agent_y]

    def set_target(self, target_position):
        img = PIL.Image.open("../figures/pngegg.png")

        resized_image = img.resize((self.line_len-10, self.line_len-10))
        self.target_image = PIL.ImageTk.PhotoImage(resized_image, master=self.window)

        self.target_y = int(20 + self.line_len * target_position[0] + self.line_len/2)
        self.target_x = int(20 + self.line_len * target_position[1] + self.line_len/2)
        self.canvas.create_image(self.target_x, self.target_y, anchor=CENTER, image=self.target_image)
