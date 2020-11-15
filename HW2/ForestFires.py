import numpy as np
from matplotlib import pyplot as plt
import numpy.random as rnd
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

        plt.clf()
        plt.scatter(tree_coordinates_x, tree_coordinates_y, color='green', label='Trees')
        plt.scatter(fire_coordinates_x, fire_coordinates_y, color='orange', label='Fire')
        plt.axis([-2, self.N + 2, -2, self.N + 2])
        plt.title('Forest Fire')
        plt.draw()
        if len(self.fire) > 0:
            plt.pause(0.5)
        else:
            plt.pause(0.00001)

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
        found_pos = False
        while not found_pos:
            if self.trees[i, j] == 0:
                self.trees[i, j] = 1
                found_pos = True
            else:
                (i, j) = self.random_position()

    def burn_tree(self, i, j):
        if self.trees[i, j] == 1:
            self.trees[i, j] = 0
            self.fire[i, j] = 1
    
    def burn_neighbours(self, i, j):
        
        if i > 0:
            if self.trees[i-1, j] == 1:
                self.trees[i-1, j] = 0
                self.fire[i-1, j] = 1
        
        if i < self.N - 1:
            if self.trees[i+1, j] == 1:
                self.trees[i+1, j] = 0
                self.fire[i+1, j] = 1
        
        if j < self.N - 1:
            if self.trees[i, j+1] == 1:
                self.trees[i, j+1] = 0
                self.fire[i, j+1] = 1
        
        if j > 0:
            if self.trees[i, j-1] == 1:
                self.trees[i, j-1] = 0
                self.fire[i, j-1] = 1

    def probabilities(self, q):
        tree_prob = self.tree_probability()
        fire_prob = self.fire_probability()
        q.put([tree_prob, fire_prob])

    def random_position(self):
        x = rnd.randint(0, self.N)
        y = rnd.randint(0, self.N)
        return (x, y)

    def step(self):
        (i, j) = self.random_position()
        should_grow = self.tree_probability()
        should_burn = self.fire_probability()
        if should_grow:
            self.grow_tree(i, j)
        (i, j) = self.random_position()
        if should_burn:
            self.burn_tree(i, j)
            done = False
            if np.sum(self.fire) > 0:
                while not done:
                    burning_trees_x = np.where(self.fire == 1)[0]
                    burning_trees_y = np.where(self.fire == 1)[1]
                    for i in range(len(burning_trees_x)):
                        self.burn_neighbours(burning_trees_x[i], burning_trees_y[i])            
                    burning_trees_next_x = np.where(self.fire == 1)[0]
                    if np.sum(burning_trees_x) == np.sum(burning_trees_next_x):
                        done = True

        self.draw_forest_fire()
        self.fire[self.fire == 1] = 0

    def cpu_step(self):
        queues = list()
        threads = list()

        for process in range(int(self.N / self.CPU_THREADS)):
            for i in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                id0 = 0
                id1 = 0
                id2 = 0
                for j in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                    
                    queues.append(Queue())
                    threads.append(Process(target=self.probabilities, args=(queues[id0],)))
                    threads[id0].start()
                    id0 += 1
                for j in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                    threads[id1].join()
                    id1 += 1
                for j in range(process * self.CPU_THREADS, self.CPU_THREADS + process * self.CPU_THREADS):
                    probs = queues[id2].get()
                    prob1 = probs[0]
                    prob2 = probs[1]
                    if prob1:
                        self.grow_tree(i, j)
                    if prob2:
                        self.burn_tree(i, j)
                    if self.fire[i, j] == 1:
                        self.burn_neighbours(i, j)
                    id2 += 1

                queues.clear()
                threads.clear()
                    
        self.draw_forest_fire()
                    
def task1():
    SOC = ForestFires(N=128, p=1, f=1)
    while(1):
        SOC.step()
                
task1()