from tkinter import *
import random
import PIL.Image, PIL.ImageTk

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
        self.draw_line()
        self.draw_ex_wall()
        self.draw_in_wall()

    def create_canvas(self):
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = Canvas(self.window, width = self.canvas_width, height = self.canvas_height, bg = "black", bd = 2)
        self.canvas.pack(fill = "both", expand = True)

    def draw_line(self):
        self.line_len = int((self.canvas_width - 40) / self.grid_dim)
        self.lines = []
        row = 0
        for line_num in range(self.grid_dim): # 세로 라인 생성
            col = 0
            for start_line in range(20, self.canvas_width - 20 + 1, self.line_len):
                self.canvas.create_line(start_line, (line_num*self.line_len+20), start_line, (line_num*self.line_len+20) + self.line_len, fill = "white")
                self.lines.append({'index': [row, col],
                                   'polygon_type': 'line',
                                   'dirt': 'height',
                                   'start_x': start_line,
                                   'start_y': line_num*self.line_len+20,
                                   'end_x': start_line,
                                   'end_y': (line_num*self.line_len+20) + self.line_len,
                                   'mid_coordinate': [start_line, start_line + int(self.line_len / 2)],
                                   'length': self.line_len,
                                   'outline': 'white'})
                col += 1
            row += 1
        row = 0
        for line_num in range(self.grid_dim + 1): # 가로 라인 생성
            col = 0
            for start_line in range(20, self.canvas_width - 20 - self.line_len + 1, self.line_len):
                self.canvas.create_line(start_line, (line_num*self.line_len+20), start_line + self.line_len, (line_num*self.line_len+20), fill="white")
                self.lines.append({'index': [row, col],
                                   'polygon_type': 'line',
                                   'dirt': 'width',
                                   'start_x': start_line,
                                   'start_y': line_num*self.line_len+20,
                                   'end_x': start_line + self.line_len,
                                   'end_y': (line_num*self.line_len+20),
                                   'mid_coordinate': [start_line + int(self.line_len / 2), line_num*self.line_len + 20],
                                   'length': self.line_len,
                                   'outline': 'white'})
                col += 1
            row += 1

    def draw_ex_wall(self):
        # 외벽 생성
        for col in range(self.grid_dim + 1):
            for line in self.lines:
                if (line['index'] == [0, col]) and (line['dirt'] == 'width'):
                    self.canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                                       line['end_y'], fill="blue", width=5)
                    line['outline'] = 'blue'
                if (line['index'] == [self.grid_dim, col]) and (line['dirt'] == 'width'):
                    self.canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                                       line['end_y'], fill="blue", width=5)
                    line['outline'] = 'blue'

        for row in range(self.grid_dim + 1):
            for line in self.lines:
                if (line['index'] == [row, 0]) and (line['dirt'] == 'height'):
                    self.canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                                       line['end_y'], fill="blue", width=5)
                    line['outline'] = 'blue'
                if (line['index'] == [row, self.grid_dim]) and (line['dirt'] == 'height'):
                    self.canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                                       line['end_y'], fill="blue", width=5)
                    line['outline'] = 'blue'

    def draw_in_wall(self):
        # 내벽 생성 / 칸을 막는 경우에 대한 예외처리 필요
        self.walls = []
        for _ in range(int((self.grid_dim**2 * 2 - self.grid_dim*2) * 0.1)):
            rand_row = random.randrange(1, self.grid_dim)
            rand_col = random.randrange(1, self.grid_dim)
            rand_dirt = random.choice(['width', 'height'])
            for line in self.lines:
                if (line['index'] == [rand_row, rand_col]) and (line['dirt'] == rand_dirt) and (line['outline'] == 'white'):
                    self.canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                                       line['end_y'], fill="blue", width=5)
                    line['outline'] = 'blue'
                    self.walls.append(line)

    def wall_lines(self):
        return self.walls

    def set_agent(self, agent, cardinal_point):
        if cardinal_point == "east":
            img = PIL.Image.open("pngegg2.png")
        elif cardinal_point == "west":
            img = PIL.Image.open("pngegg2.png").transpose(PIL.Image.ROTATE_180)
        elif cardinal_point == "south":
            img = PIL.Image.open("pngegg2.png").transpose(PIL.Image.ROTATE_270)
        elif cardinal_point == "north":
            img = PIL.Image.open("pngegg2.png").transpose(PIL.Image.ROTATE_90)

        resized_image = img.resize((self.line_len-10, self.line_len-10))
        self.agent_image = PIL.ImageTk.PhotoImage(resized_image, master=self.window)

        self.agent_y = int(20 + self.line_len * agent.position[0] + self.line_len/2)
        self.agent_x = int(20 + self.line_len * agent.position[1] + self.line_len/2)
        self.canvas.create_image(self.agent_x, self.agent_y, anchor=CENTER, image=self.agent_image)

    def agent_coordinate(self):
        return [self.agent_x, self.agent_y]

    def set_target(self, target_position):
        img = PIL.Image.open("pngegg.png")

        resized_image = img.resize((self.line_len-10, self.line_len-10))
        self.target_image = PIL.ImageTk.PhotoImage(resized_image, master=self.window)

        self.target_y = int(20 + self.line_len * target_position[0] + self.line_len/2)
        self.target_x = int(20 + self.line_len * target_position[1] + self.line_len/2)
        self.canvas.create_image(self.target_x, self.target_y, anchor=CENTER, image=self.target_image)