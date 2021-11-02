import time
from pacman_entity import *
from simulation_entity import *
from TD_onestep import *
import threading

class World:
    def __init__(self, n):
        self.grid_dim = n;
        self.env = Env(n)
        self.pacman = Pacman(n)
        print("Initialize")
        self.pacman.visualization()

        print("-------------------------------")
        input("press any key")

        self.step = 0
        self.pacman_action_list = ["straight", "left", "right"]
        self.pacman_cardinal_points = ["north", "east", "south", "west"]

        self.model = TDZero(self)
        self.thread = threading.Thread(target=self.model.iteration)

    def main(self):

        self.window = WindowTkinter().create_window()
        self.cv = CanvasGrid(self.window, self.pacman, self.env.walls)
        self.cv.set_agent(self.pacman, self.pacman.cardinal_point)
        self.cv.set_target(self.env.gridmap_goal)

        self.thread.daemon = True
        self.thread.start()
        # self.cv.canvas.bind_all("<Key>", self.iter_step)

        self.window.mainloop()

        print(self.model.Q.reshape(-1, 3))
        print(self.model.policy_matrix)
        print("Pacman arrived at goal in {} steps.".format(self.step))


    def iter_step(self):
        self.step = 0
        self.pacman.position = self.pacman.first_position
        while True:
            # if event.keysym and np.any(self.pacman.gridmap_goal != self.pacman.position):
            #print("-------------------------------")
            #print("step: ", self.step)
            #pacman_direction = random.choice(self.pacman_action_list)
            pacman_cardinal_point = self.pacman_cardinal_points.index(self.pacman.cardinal_point)
            pacman_direction = np.random.choice(self.pacman_action_list, 1, p=self.model.policy_matrix[
                self.pacman.position[0] * (4 * 5) + self.pacman.position[1] * 4 + pacman_cardinal_point])

            self.step += 1

            if pacman_direction == "straight":
                if self.pacman.straight(self.env.walls) == 1:
                    self.step += 1
                    print('step = ', self.step)
                    return next_s
                #self.pacman.visualization()
            elif pacman_direction == "left":
                self.pacman.left()
                #self.pacman.visualization()
            elif pacman_direction == "right":
                self.pacman.right()
                #self.pacman.visualization()
            self.cv.set_agent(self.pacman, self.pacman.cardinal_point)
            #time.sleep(0.3)

            pacman_state = np.append(self.pacman.position,
                                     self.pacman_cardinal_points.index(self.pacman.cardinal_point))
            next_s = pacman_state
            return next_s



if __name__ == '__main__':
    World(5).main()
