import time
from scripts.packman_entity import *
from scripts.simulation_entity import *
import threading

class World:
    def __init__(self):
        n = int(input("Input grid size N:"))

        # self.env = Env(n)
        self.pacman = Pacman(n)
        gridmap = self.pacman.gridmap
        print("Initialize")
        self.pacman.visualization()
        self.thread = threading.Thread(target=self.iter_step)
        print("-------------------------------")
        input("press any key")

        self.step = 0
        self.pacman_action_list = ["straight", "left", "right"]

    def main(self):
        self.window = WindowTkinter().create_window()
        self.cv = CanvasGrid(self.window, self.pacman)
        self.cv.set_agent(self.pacman, self.pacman.cardinal_point)
        self.cv.set_target(self.pacman.goal_position())

        self.thread.daemon = True
        self.thread.start()
        # self.cv.canvas.bind_all("<Key>", self.iter_step)

        self.window.mainloop()

        print("Pacman arrived at goal in {} steps.".format(self.step))


    def iter_step(self):
        while True:
            # if event.keysym and np.any(self.pacman.gridmap_goal != self.pacman.position):
            print("-------------------------------")
            self.step += 1
            print("step: ", self.step)
            # pacman_direction = random.choice(pacman_action_list)
            pacman_direction = np.random.choice(self.pacman_action_list, 1, p=[0.8,0.1,0.1])
            if pacman_direction == "straight":
                if self.pacman.straight(self.cv.wall_lines(), self.cv.agent_coordinate()) == 1:
                    self.window.destroy()
                self.pacman.visualization()
            elif pacman_direction == "left":
                self.pacman.left()
                self.pacman.visualization()
            elif pacman_direction == "right":
                self.pacman.right()
                self.pacman.visualization()
            self.cv.set_agent(self.pacman, self.pacman.cardinal_point)
            time.sleep(1)


if __name__ == '__main__':
    try:
        World().main()
    except ValueError:
        print("grid size N > 1")