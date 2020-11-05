import random as rnd
import matplotlib.pyplot as plt
import numpy as np

class DiseaseSpreading:
    def __init__(self, nr_agents, grid_size, diffusion_rate):
        self.x = np.random.randint(0, 100, nr_agents)
        self.y = np.random.randint(0, 100, nr_agents)
        self. nr_agents = nr_agents
        self.grid_size = grid_size
        self.d = diffusion_rate

    def draw_agents(self):
        plt.scatter(self.x, self.y)
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    def neumann_probability(self):
        r = rnd.uniform(0, 1)
        if r <= self.d:
            return True
        return False

    def move_agent(self, i):
        found_a_way = False
        while(found_a_way == False):
            r = rnd.randint(0, 4)
            if r == 0 and self.x[i] < 100:
                self.x[i] += 1
                found_a_way = True
            elif r == 1 and self.x[i] > 0:
                self.x[i] -= 1
                found_a_way = True
            elif r == 2 and self.y[i] < 100:
                self.y[i] += 1
                found_a_way = True
            elif r == 3 and self.y[i] > 0:
                self.y[i] -= 1
                found_a_way = True

    def move_agents(self):
        agent_should_move = self.neumann_probability()
        if agent_should_move:
            for i in range(self.nr_agents):
                self.move_agent(i)
                

def main():
    dis = DiseaseSpreading(nr_agents=1000, grid_size=100, diffusion_rate=0.1)
    while(1):
        dis.draw_agents()
        dis.move_agents()

main()