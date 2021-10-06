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
    def __init__(self, window, pacman):
        self.window = window
        self.pacman = pacman
        self.grid_dim = self.pacman.n
        # self.window = Tk()
        self.run()

    def run(self):
        self.create_canvas()
        self.walls = self.pacman.wall_lines()
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

    def set_pacman(self, pacman, cardinal_point):
        if cardinal_point == 1:
            img = PIL.Image.open("../figures/pngegg2.png")
        elif cardinal_point == 3:
            img = PIL.Image.open("../figures/pngegg2.png").transpose(PIL.Image.ROTATE_180)
        elif cardinal_point == 2:
            img = PIL.Image.open("../figures/pngegg2.png").transpose(PIL.Image.ROTATE_270)
        elif cardinal_point == 0:
            img = PIL.Image.open("../figures/pngegg2.png").transpose(PIL.Image.ROTATE_90)

        resized_image = img.resize((self.line_len-10, self.line_len-10))
        self.pacman_image = PIL.ImageTk.PhotoImage(resized_image, master=self.window)

        self.pacman_y = int(20 + self.line_len * pacman.position[0] + self.line_len/2)
        self.pacman_x = int(20 + self.line_len * pacman.position[1] + self.line_len/2)
        self.canvas.create_image(self.pacman_x, self.pacman_y, anchor=CENTER, image=self.pacman_image)

    def pacman_coordinate(self):
        return [self.pacman_x, self.pacman_y]

    def set_target(self, target_position):
        img = PIL.Image.open("../figures/pngegg.png")

        resized_image = img.resize((self.line_len-10, self.line_len-10))
        self.target_image = PIL.ImageTk.PhotoImage(resized_image, master=self.window)

        self.target_y = int(20 + self.line_len * target_position[0] + self.line_len/2)
        self.target_x = int(20 + self.line_len * target_position[1] + self.line_len/2)
        self.canvas.create_image(self.target_x, self.target_y, anchor=CENTER, image=self.target_image)
