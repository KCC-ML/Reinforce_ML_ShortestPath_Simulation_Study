from tkinter import *

window = Tk()
window.title("grid world")
window.geometry("500x500")
window.resizable(True, True)

n = int(input())
grid_dim = n

canvas_width = canvas_height = 500
canvas = Canvas(window, width = canvas_width, height = canvas_height, bg = "black", bd = 2)
canvas.pack(fill = "both", expand = True)

line_len = int((canvas_width - 40) / grid_dim)
lines = []
row = 0; col = 0
for line_num in range(grid_dim): # 세로 라인 생성
    for start_line in range(20, canvas_width - 20 + 1, line_len):
        canvas.create_line(start_line, (line_num*line_len+20), start_line, (line_num*line_len+20) + line_len, fill = "white")
        lines.append({'index': [row, col],
                       'polygon_type': 'line',
                      'role': 'base',
                       'start_x': start_line,
                       'start_y': line_num*line_len+20,
                       'end_x': start_line,
                       'end_y': (line_num*line_len+20) + line_len,
                       'outline': 'white'})
        col += 1
    row += 1
row = 0; col = 0
for line_num in range(grid_dim + 1): # 가로 라인 생성
    for start_line in range(20, canvas_width - 20 - line_len + 1, line_len):
        canvas.create_line(start_line, (line_num*line_len+20), start_line + line_len, (line_num*line_len+20), fill="white")
        lines.append({'index': [row, col],
                       'polygon_type': 'line',
                      'role': 'base',
                       'start_x': start_line,
                       'start_y': line_num*line_len+20,
                       'end_x': start_line,
                       'end_y': (line_num*line_len+20) + line_len,
                       'outline': 'white'})
        col += 1
    row += 1

# 외벽 생성


window.mainloop()




# rectangles = []
# rect_size = int((canvas_width - 40) / grid_dim)
# row = 0; col = 0
# for rect_y in range(20, canvas_width-20-rect_size+1, rect_size):
#     for rect_x in range(20, canvas_width-20-rect_size+1, rect_size):
#         canvas.create_rectangle(rect_x, rect_y, rect_x+rect_size, rect_y+rect_size, outline="white", width=2)
#         rectangles.append({'index': [row, col],
#                            'polygon_type': 'rectangle',
#                            'start_x': rect_x,
#                            'start_y': rect_y,
#                            'end_x': rect_x+rect_size,
#                            'end_y': rect_y+rect_size,
#                            'outline': 'white'})
#         col += 1
#     row += 1