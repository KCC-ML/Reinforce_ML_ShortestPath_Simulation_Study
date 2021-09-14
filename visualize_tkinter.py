from tkinter import *
import random
from packman import *
from PIL import Image,ImageTk
import numpy as np

window = Tk()
window.title("grid world")
window.geometry("500x500")
window.resizable(True, True)

n = int(input())
# grid_dim = n

# 에이전트 삽입
env = Env(n)
pacman = Pacman(n)
gridmap = pacman.gridmap
print(gridmap)
print(pacman.position)
grid_dim = n


# 시각화
canvas_width = canvas_height = 500
canvas = Canvas(window, width = canvas_width, height = canvas_height, bg = "black", bd = 2)
canvas.pack(fill = "both", expand = True)

line_len = int((canvas_width - 40) / grid_dim)
lines = []
row = 0
for line_num in range(grid_dim): # 세로 라인 생성
    col = 0
    for start_line in range(20, canvas_width - 20 + 1, line_len):
        canvas.create_line(start_line, (line_num*line_len+20), start_line, (line_num*line_len+20) + line_len, fill = "white")
        lines.append({'index': [row, col],
                       'polygon_type': 'line',
                      'dirt': 'height',
                       'start_x': start_line,
                       'start_y': line_num*line_len+20,
                       'end_x': start_line,
                       'end_y': (line_num*line_len+20) + line_len,
                       'outline': 'white'})
        col += 1
    row += 1
row = 0
for line_num in range(grid_dim + 1): # 가로 라인 생성
    col = 0
    for start_line in range(20, canvas_width - 20 - line_len + 1, line_len):
        canvas.create_line(start_line, (line_num*line_len+20), start_line + line_len, (line_num*line_len+20), fill="white")
        lines.append({'index': [row, col],
                       'polygon_type': 'line',
                      'dirt': 'width',
                       'start_x': start_line,
                       'start_y': line_num*line_len+20,
                       'end_x': start_line + line_len,
                       'end_y': (line_num*line_len+20),
                       'outline': 'white'})
        col += 1
    row += 1

# 외벽 생성
print(lines)
for col in range(grid_dim + 1):
    for line in lines:
        if (line['index'] == [0, col]) and (line['dirt'] == 'width'):
            canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                               line['end_y'], fill="red")
            line['outline'] = 'red'
        if (line['index'] == [grid_dim, col]) and (line['dirt'] == 'width'):
            canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                               line['end_y'], fill="red")
            line['outline'] = 'red'

for row in range(grid_dim + 1):
    for line in lines:
        if (line['index'] == [row, 0]) and (line['dirt'] == 'height'):
            canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                               line['end_y'], fill="red")
            line['outline'] = 'red'
        if (line['index'] == [row, grid_dim]) and (line['dirt'] == 'height'):
            canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                               line['end_y'], fill="red")
            line['outline'] = 'red'

# 내벽 생성 / 칸을 막는 경우에 대한 예외처리 필요
for _ in range(int(grid_dim**2 * 0.1)):
    rand_row = random.randrange(1, grid_dim)
    rand_col = random.randrange(1, grid_dim)
    rand_dirt = random.choice(['width', 'height'])
    for line in lines:
        if (line['index'] == [rand_row, rand_col]) and (line['dirt'] == rand_dirt) and (line['outline'] == 'white'):
            canvas.create_line(line['start_x'], line['start_y'], line['end_x'],
                               line['end_y'], fill="red")
            line['outline'] = 'red'


# ndarray와 좌표 연동
#Load an image in the script
img= (Image.open("pngegg.png"))

#Resize the Image using resize method
resized_image = img.resize((line_len-5, line_len-5))
new_image = ImageTk.PhotoImage(resized_image)

#Add image to the Canvas Items
# 칸의 왼쪽 위 꼭짓점 좌표에 대하여 gridmap에서의 pacman 좌표를 표시
pacman_x = int(20 + line_len * pacman.position[0])
pacman_y = int(20 + line_len * pacman.position[1])
canvas.create_image(pacman_x, pacman_y, anchor=NW, image=new_image)

# pacman 이동
pacman.set_packman()
pacman_x = int(20 + line_len * pacman.position[0])
pacman_y = int(20 + line_len * pacman.position[1])
canvas.create_image(pacman_x, pacman_y, anchor=NW, image=new_image)

window.mainloop()