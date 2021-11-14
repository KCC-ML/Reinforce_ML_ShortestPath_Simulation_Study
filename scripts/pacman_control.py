import time
from pacman_entity import *
from simulation_entity import *
from SARSA_lambda_backward import *
import threading
import matplotlib.pyplot as plt

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

        self.model = SARSALB(self)
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

        #print(self.model.V.reshape(-1, 4))
        #print(self.model.policy_matrix)
        #print("Pacman arrived at goal in {} steps.".format(self.step))
        plt.plot(self.model.steps)
        #plt.ylim(0, 200)
        plt.show()

    def iter_step(self, pacman_direction):
        #self.step = 0
        #self.pacman.position = self.pacman.first_position

        # if event.keysym and np.any(self.pacman.gridmap_goal != self.pacman.position):
        #print("-------------------------------")
        #print("step: ", self.step)
        #pacman_direction = random.choice(self.pacman_action_list)


        #self.step += 1

        if pacman_direction == "straight":
            if self.pacman.straight(self.env.walls) == 1:
                #self.step += 1
                #print('step = ', self.step)
                pacman_state = np.append(self.pacman.position,
                                         self.pacman_cardinal_points.index(self.pacman.cardinal_point))
                reward = 5
                return pacman_state, reward
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
        reward = -1
        return pacman_state, reward


if __name__ == '__main__':
    World(5).main()
