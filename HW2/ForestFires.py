import numpy as np
from matplotlib import pyplot as plt
import random as rnd
import multiprocessing
from multiprocessing import Process, Queue

class ForestFires:
    def __init__(self, N, p, f):
        self.N = N
        self.p = p
        self.f = f
        self.trees = np.zeros((self.N, self.N))
        self.fire = np.zeros((self.N, self.N))
        self.CPU_THREADS = multiprocessing.cpu_count()
    
    def draw_forest_fire(self):

        tree_coordinates_x = np.where(self.trees == 1)[0]
        tree_coordinates_y = np.where(self.trees == 1)[1]
        fire_coordinates_x = np.where(self.fire == 1)[0]
        fire_coordinates_y = np.where(self.fire == 1)[1]

        plt.scatter(tree_coordinates_x, tree_coordinates_y, color='green', label='Trees')
        plt.scatter(fire_coordinates_x, fire_coordinates_y, color='orange', label='Fire')
        plt.axis([-2, self.N + 2, -2, self.N + 2])
        plt.title('Forest Fire')
        plt.draw()
        plt.pause(1)

    def tree_probability(self):
        r = rnd.uniform(0, 1)
        if r <= self.p:
            return True
        return False
    
    def fire_probability(self):
        r = rnd.uniform(0, 1)
        if r <= self.f:
            return True
        return False

    def grow_tree(self, i, j):
        if self.trees[i, j] == 0:
            self.trees[i, j] = 1

    def burn_tree(self, i, j):
        if self.fire[i, j] == 0:
            self.fire[i, j] = 1
    
    def burn_neighbours(self, i, j):
        
        if self.trees[i+1, j] == 1:
            self.fire[i+1, j] = 1

        if self.trees[i-1, j] == 1:
            self.fire[i-1, j] = 1

        if self.trees[i, j+1] == 1:
            self.fire[i, j+1] = 1

        if self.trees[i, j-1] == 1:
            self.fire[i, j-1] = 1

    def probabilities(self, q):
        q.put([self.tree_probability(), self.fire_probability, self.neumann_probability])

    def step(self):
        queues = list()
        threads = list()

        for process in range(int(self.N / self.CPU_THREADS)):
            for i in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                for j in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                    queues.append(Queue())
                    threads.append(Process(target=self.probabilities, args=(queues[j])))
                    threads[j].start()
                for j in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                    threads[j].join()
                for j in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                    if queues[j][0]:
                        self.grow_tree(i, j)
                    if queues[j][1]:
                        self.burn_tree(i, j)
                    if self.fire[i, j] == 1:
                        self.burn_neighbours(i, j)
                    
def task1():
    SOC = ForestFires(N=128, p=0.5, f=0.01)
    while(1):
        SOC.step()
        SOC.draw_forest_fire()
                
task1()