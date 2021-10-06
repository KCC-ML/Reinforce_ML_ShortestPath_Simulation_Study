import time
from scripts.pacman_entity import *
from scripts.simulation_entity import *
from scripts.MDP import *
import threading
from scripts.MC_prediction import *

class World:
    def __init__(self, n):
        # self.env = Env(n)
        self.pacman = Pacman(n)
        gridmap = self.pacman.gridmap
        print("Initialize")
        self.pacman.visualization()
        self.thread = threading.Thread(target=self.iter_step)
        print("-------------------------------")
        # input("press any key")
        self.algorithm = 'MCP' # ['random', 'MDP', 'MCP']

        self.step = 0
        self.pacman_action_list = [0, 1, 2]


    def main(self):
        self.window = WindowTkinter().create_window()
        self.cv = CanvasGrid(self.window, self.pacman)
        self.cv.set_pacman(self.pacman, self.pacman.cardinal_point)
        self.cv.set_target(self.pacman.goal_position())

        if self.algorithm == 'MDP':
            self.MDP = MDP(self.cv)
            value_function = self.MDP.policy_evaluation(100)
            self.policy = self.MDP.policy_improvement(value_function)
        elif self.algorithm == 'MCP':
            self.policy = MC_prediction(self.pacman, 1000).optimal_policy()

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

            if self.algorithm == 'random':
                # pacman_direction = random.choice(self.pacman_action_list)
                pacman_direction = np.random.choice(self.pacman_action_list, 1, p=[0.8,0.1,0.1])
            elif self.algorithm == 'MDP':
                tmp = 4 * ((self.pacman.n * self.pacman.position[0]) + self.pacman.position[
                    1]) + self.pacman.cardinal_point
                pacman_direction = np.random.choice(self.pacman_action_list, 1, p=self.policy[tmp])
            elif self.algorithm == 'MCP':
                tmp = 4 * ((self.pacman.n * self.pacman.position[0]) + self.pacman.position[
                    1]) + self.pacman.cardinal_point
                pacman_direction = self.policy[tmp]

            if pacman_direction == 0:
                # pacman gets target
                if self.pacman.straight() == 1:
                    self.window.destroy()
                self.pacman.visualization()
            elif pacman_direction == 1:
                self.pacman.left()
                self.pacman.visualization()
            elif pacman_direction == 2:
                self.pacman.right()
                self.pacman.visualization()
            self.cv.set_pacman(self.pacman, self.pacman.cardinal_point)
            time.sleep(1)


if __name__ == '__main__':
        World(5).main()