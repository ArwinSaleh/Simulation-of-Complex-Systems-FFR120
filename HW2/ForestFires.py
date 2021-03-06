import numpy as np
from matplotlib import pyplot as plt
import numpy.random as rnd
import multiprocessing
from multiprocessing import Process, Queue
from numpy import asarray
from numpy import savetxt
from numpy import genfromtxt

class ForestFires:
    def __init__(self, N, p, f, GLOBE=True, SNAPSHOT=False):
        self.N = N
        self.p = p
        self.f = f
        self.trees = np.zeros((self.N, self.N))
        self.fire = np.zeros((self.N, self.N))
        self.CPU_THREADS = multiprocessing.cpu_count()
        self.burned_cluster_data = list()
        self.total_trees_data = list()
        self.trees_data = 0
        self.fires_data = 0
        self.GLOBE = GLOBE
        self.SNAPSHOT = SNAPSHOT
        self.time_step = 1

        # Task 2 helper variables
        self.burned_cluster_data_ALT = list()
        self.total_trees_data_ALT = list()

    def initialize_trees(self, DENSITY=0.75):
        for i in range(int(DENSITY * self.N ** 2)):
            (x, y) = self.random_position()
            self.grow_tree(x, y)
    def draw_forest_fire(self):

        tree_coordinates_x = np.where(self.trees == 1)[0]
        tree_coordinates_y = np.where(self.trees == 1)[1]
        fire_coordinates_x = np.where(self.fire == 1)[0]
        fire_coordinates_y = np.where(self.fire == 1)[1]

        plt.clf()
        plt.scatter(tree_coordinates_x, tree_coordinates_y, color='green', label='Trees')
        plt.scatter(fire_coordinates_x, fire_coordinates_y, color='orange', label='Fire')
        plt.axis([-2, self.N + 2, -2, self.N + 2])
        plt.title('p = ' + str(self.p) + '      f = ' + str(self.f) + '         p/f = ' + str(self.p / self.f))
        plt.draw()
        if np.any(self.fire == 1) > 0:
            if self.SNAPSHOT:
                plt.savefig('burned_cluster_step' + str(self.time_step))
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
            self.burn_tree(i-1, j)
        else:
            if self.GLOBE:
                self.burn_tree(self.N-1, j)
        if i < self.N - 1:
            self.burn_tree(i+1, j)
        else:
            if self.GLOBE:
                self.burn_tree(0, j)
        if j < self.N - 1:
            self.burn_tree(i, j+1)
        else:
            if self.GLOBE:
                self.burn_tree(i, 0)
        if j > 0:
            self.burn_tree(i, j-1)
        else:
            if self.GLOBE:
                self.burn_tree(i, self.N-1)

    def probabilities(self, q):
        tree_prob = self.tree_probability()
        fire_prob = self.fire_probability()
        q.put([tree_prob, fire_prob])

    def random_position(self):
        x = rnd.randint(0, self.N)
        y = rnd.randint(0, self.N)
        return (x, y)

    def step(self, DRAW=True):
        empty_sites_x = np.where(self.trees == 0)[0]
        empty_sites_y = np.where(self.trees == 0)[1]
        for empty_site in range(len(empty_sites_x)):
            should_grow = self.tree_probability()
            if should_grow:
                self.grow_tree(empty_sites_x[empty_site], empty_sites_y[empty_site])
        (i, j) = self.random_position()
        should_burn = self.fire_probability()
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
                        self.burned_cluster_data.append(np.sum(self.fire))
                        self.total_trees_data.append(np.sum(self.trees))
                        print("BURNED CLUSTER SIZE: " + str(self.burned_cluster_data) + "\n\nTOTAL NUMBER OF TREES: " + str(self.total_trees_data))
                        done = True
        
    def step_size_dist(self, DRAW=True):

        empty_sites_x = np.where(self.trees == 0)[0]
        empty_sites_y = np.where(self.trees == 0)[1]
        for empty_site in range(len(empty_sites_x)):
            should_grow = self.tree_probability()
            if should_grow:
                self.grow_tree(empty_sites_x[empty_site], empty_sites_y[empty_site])
        (i, j) = self.random_position()
        should_burn = self.fire_probability()
        if should_burn:
            self.burn_tree(i, j)
            done = False
            if np.sum(self.fire) > 0:
                forest_density = np.sum(self.trees) / self.N
                while not done:
                    burning_trees_x = np.where(self.fire == 1)[0]
                    burning_trees_y = np.where(self.fire == 1)[1]
                    for i in range(len(burning_trees_x)):
                        self.burn_neighbours(burning_trees_x[i], burning_trees_y[i])            
                    burning_trees_next_x = np.where(self.fire == 1)[0]
                    if np.sum(burning_trees_x) == np.sum(burning_trees_next_x):
                        self.burned_cluster_data.append(np.sum(self.fire))
                        self.total_trees_data.append(np.sum(self.trees))
                        print("BURNED CLUSTER SIZE: " + str(self.burned_cluster_data) + "\n\nTOTAL NUMBER OF TREES: " + str(self.total_trees_data))
                        done = True
                self.fire[self.fire == 1] = 0
                self.initialize_trees(DENSITY=forest_density)
                while(np.sum(self.fire) == 0):
                    (x, y) = self.random_position()
                    self.burn_tree(x, y)
                print("TEST")
                done = False
                while not done:
                    burning_trees_x = np.where(self.fire == 1)[0]
                    burning_trees_y = np.where(self.fire == 1)[1]
                    for i in range(len(burning_trees_x)):
                        self.burn_neighbours(burning_trees_x[i], burning_trees_y[i])            
                    burning_trees_next_x = np.where(self.fire == 1)[0]
                    if np.sum(burning_trees_x) == np.sum(burning_trees_next_x):
                        self.burned_cluster_data_ALT.append(np.sum(self.fire))
                        self.total_trees_data_ALT.append(np.sum(self.trees))
                        print("BURNED CLUSTER SIZE: " + str(self.burned_cluster_data) + "\n\nTOTAL NUMBER OF TREES: " + str(self.total_trees_data))
                        done = True
                

        if DRAW:
            self.draw_forest_fire()
        self.fire[self.fire == 1] = 0

    def cpu_step(self):     # Currently not working
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
    SOC = ForestFires(N=128, p=0.001, f=0.001, GLOBE=True, SNAPSHOT=True)

    for i in range(10000):
        SOC.step(DRAW=True)
        print("TIME STEP: " + str(i + 1))
        SOC.time_step += 1
                
def task2_NOT_INIT():
    SOC = ForestFires(N=128, p=0.001, f=0.05, GLOBE=True, SNAPSHOT=False)

    for i in range(10000):
        SOC.step(DRAW=False)
        print("TIME STEP: " + str(i + 1))
        SOC.time_step += 1

    # Sort data
    SOC.total_trees_data = sorted(SOC.total_trees_data, reverse=True)
    SOC.burned_cluster_data = sorted(SOC.burned_cluster_data, reverse=True)

    savetxt("trees_p" + str(SOC.p) + "_f" + str(SOC.f) + "_NOT_INIT.csv", asarray(SOC.total_trees_data), delimiter=',')
    savetxt("fires_p" + str(SOC.p) + "_f" + str(SOC.f) + "_NOT_INIT.csv", asarray(SOC.burned_cluster_data), delimiter=',')

def task2_INIT():
    SOC = ForestFires(N=128, p=0.001, f=0.05, GLOBE=True, SNAPSHOT=False)
    SOC.initialize_trees(PERCENTAGE=0.25)
    for i in range(10000):
        SOC.step(DRAW=False)
        print("TIME STEP: " + str(i + 1))
        SOC.time_step += 1

    # Sort data
    SOC.total_trees_data = sorted(SOC.total_trees_data, reverse=True)
    SOC.burned_cluster_data = sorted(SOC.burned_cluster_data, reverse=True)

    savetxt("trees_p" + str(SOC.p) + "_f" + str(SOC.f) + "_INIT.csv", asarray(SOC.total_trees_data), delimiter=',')
    savetxt("fires_p" + str(SOC.p) + "_f" + str(SOC.f) + "_INIT.csv", asarray(SOC.burned_cluster_data), delimiter=',')

def task2():
    SOC = ForestFires(N=128, p=0.001, f=1, GLOBE=True, SNAPSHOT=False)
    for i in range(10000):
        SOC.step_size_dist(DRAW=False)
        print("TIME STEP: " + str(i + 1))
        SOC.time_step += 1

    # Sort data
    SOC.total_trees_data = sorted(SOC.total_trees_data, reverse=True)
    SOC.burned_cluster_data = sorted(SOC.burned_cluster_data, reverse=True)

    savetxt("trees_p" + str(SOC.p) + "_f" + str(SOC.f) + "_TASK2.csv", asarray(SOC.total_trees_data), delimiter=',')
    savetxt("fires_p" + str(SOC.p) + "_f" + str(SOC.f) + "_TASK2.csv", asarray(SOC.burned_cluster_data), delimiter=',')
    savetxt("trees_p" + str(SOC.p) + "_f" + str(SOC.f) + "_TASK2_ALT.csv", asarray(SOC.total_trees_data_ALT), delimiter=',')
    savetxt("fires_p" + str(SOC.p) + "_f" + str(SOC.f) + "_TASK2_ALT.csv", asarray(SOC.burned_cluster_data_ALT), delimiter=',')

def finite_size(lattice):
    SOC = ForestFires(N=lattice, p=0.001, f=0.1, GLOBE=True, SNAPSHOT=False)
    SOC.initialize_trees()
    for i in range(5000):
        SOC.step(DRAW=False)
        print("TIME STEP: " + str(i + 1))
        SOC.time_step += 1

    # Sort data
    SOC.total_trees_data = sorted(SOC.total_trees_data, reverse=True)
    SOC.burned_cluster_data = sorted(SOC.burned_cluster_data, reverse=True)

    savetxt("N_" + str(SOC.N) + "_trees_p" + str(SOC.p) + "_f" + str(SOC.f) + ".csv", asarray(SOC.total_trees_data), delimiter=',')
    savetxt("N_" + str(SOC.N) + "_fires_p" + str(SOC.p) + "_f" + str(SOC.f) + ".csv", asarray(SOC.burned_cluster_data), delimiter=',')

def task4():
    lattices = [512]
    for lattice in lattices:
        finite_size(lattice)

#task1()
#task2_NOT_INIT()
#task2_INIT()
task2()
#task4() 