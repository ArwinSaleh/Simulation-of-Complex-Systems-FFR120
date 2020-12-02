import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sb
from scipy.stats import binom
from numpy import genfromtxt
import pandas as pd

class Network:
    def __init__(self, n=0, c=0, p=0, nr_steps=0, SELF_EDGES=False):
        self.G = nx.Graph()
        self.nr_steps = nr_steps
        self.p = p
        self.c = c
        self.n = n              # Nodes
        self.m = n * (n - 1)    # Edges
        self.adj_matrix = np.zeros((self.n, self.n))
        self.SELF_EDGES = SELF_EDGES
    
    def load_data(self):
        example = genfromtxt('HW4/data/smallWorldExample.txt', delimiter=',')
        self.adj_matrix = example

    def add_edge(self, v1, v2):
        if v1 != v2:
            self.adj_matrix[v1, v2] = 1
            self.adj_matrix[v2, v1] = 1
        
    def del_edge(self, v1, v2):
        self.adj_matrix[v1, v2] = 0
        self.adj_matrix[v2, v1] = 0

    def initialize_erdos(self):
        self.adj_matrix = np.random.uniform(0, 1, size=(self.n, self.n))
        self.adj_matrix[np.where(self.adj_matrix <= self.p)] = 1
        self.adj_matrix[np.where(self.adj_matrix != 1)] = 0
        if self.SELF_EDGES:
            np.fill_diagonal(self.adj_matrix, 2)
        else:
            np.fill_diagonal(self.adj_matrix, 0)
        
        self.m = np.sum(self.adj_matrix[np.where(self.adj_matrix==1)])
    
    def init_small_world(self):
        t = np.linspace(0, 2 * math.pi, self.n)
        x = np.cos(t)
        y = np.sin(t)

        for i in range(len(x)):
            self.G.add_node(i, pos=(x[i], y[i]))

        self.adj_matrix = np.zeros((self.n, self.n))
        for i in range(-self.n, self.n):
            for j in range(1, int(self.c / 2) + 1):
                if (i + j) > self.n - 1:
                    i = (i + j) - self.n
                    self.add_edge(i, i + j)
                else:
                    self.add_edge(i, i + j)
                    
        if self.SELF_EDGES:
            np.fill_diagonal(self.adj_matrix, 2)
        else:
            np.fill_diagonal(self.adj_matrix, 0)
        
        self.m = np.sum(self.adj_matrix[np.where(self.adj_matrix==1)])

    def add_shortcuts(self):
        for i in range(int(self.m)):
            r = np.random.uniform()
            if r < self.p:
                j = (np.random.randint(self.n), np.random.randint(self.n))
                self.adj_matrix[j[0], j[1]] = 1
        self.m = np.sum(self.adj_matrix[np.where(self.adj_matrix==1)])

    def build_graph(self):
        v1 = np.where(self.adj_matrix == 1)[0]
        v2 = np.where(self.adj_matrix == 1)[1]
        
        for i in range(len(v1)):
            self.G.add_edge(v1[i], v2[i])

    def plot_graph(self, CIRCULAR=False, STEP=False, CLUSTER=False, PATH=False):
        plt.figure()
        
        if CIRCULAR:
            plt.title('p = ' + str(self.p) + '      c = ' + str(self.c) + '      n = ' + str(self.n) + '     m = ' + str(self.m))
            nx.draw_circular(self.G)
        elif STEP:
            plt.title('p = ' + str(self.p) + '      c = ' + str(self.c) + '      n = ' + str(self.n) + '     m = ' + str(self.m) + 
            '\nNumber of steps: ' + str(self.nr_steps))
            nx.draw(self.G)
        elif CLUSTER:
            plt.title('smallWorldExample.txt' + '\nn = ' + str(self.n) + '     m = ' + str(self.m) + '\nClustering Coefficient = ' + str(self.c))
            nx.draw(self.G)
        elif PATH:
            plt.title('smallWorldExample.txt' + '\nn = ' + str(self.n) + '     m = ' + str(self.m) + '\nAverage Path Length = ' + str(self.c))
            nx.draw(self.G)
        else:
            plt.title('p = ' + str(self.p) + '      n = ' + str(self.n) + '     m = ' + str(self.m))
            nx.draw(self.G)

    
    def plot_degree_dist(self):
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        plt.figure()
        synth_data = binom.rvs(n=self.n * 2, p=self.p, loc=0, size=self.n)
        sb.distplot(synth_data, kde=True, bins=50, label='Synthetic')
        sb.distplot(sorted(degrees, reverse=True), kde=True, bins=50, label='Simulated')
        plt.xlabel('Degrees')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution' + '\np = ' + str(self.p) + '      n = ' + str(self.n))
        plt.legend()
    
    def plot_cum_ddist(self):
        d_seq = [self.G.degree(n) for n in self.G.nodes()]
        k = np.linspace(0, max(d_seq),max(d_seq))
        power_dist = 2 * (self.c ** 2) * k ** (1 - 3)

        degrees = np.array(d_seq)
        d_list = []
        for i in range(max(degrees)):
            d_list.append(np.sum(degrees == i))
        d_list = np.flip(d_list)    
        d_cdf = np.flip(np.cumsum(d_list))
        plt.figure()
        plt.loglog(k, power_dist, label='Synthetic')
        plt.loglog(k, d_cdf, linestyle='none', marker='.', label = 'Simulated')
        plt.xlabel('Degree')
        plt.ylabel('cCDF')
        plt.title('p = ' + str(self.p) + '      c = ' + str(self.c) + '      n = ' + str(self.n) + '     m = ' + str(self.m) + 
            '\nNumber of steps: ' + str(self.nr_steps))
        plt.legend()

    def compute_clustering_coefficient(self):
        adj_3 = np.dot(np.dot(self.adj_matrix, self.adj_matrix), self.adj_matrix)

        self.n = np.sum(self.adj_matrix, axis=1)
        nr_triangles = np.trace(adj_3) // 6
        nr_connected_triples = sum(self.n * (self.n - 1) / 2)
        self.n = len(self.adj_matrix)
        self.m = np.sum(self.adj_matrix[np.where(self.adj_matrix==1)])
        
        clustering_coefficient = 3 * nr_triangles / nr_connected_triples
        self.m = np.sum(self.adj_matrix[np.where(self.adj_matrix==1)])
        return round(clustering_coefficient, 6)

    def compute_path_diameter(self):
        self.n = len(self.adj_matrix)
        path_lengths = np.zeros((self.n, self.n))
        for i in range(self.n):
            path = np.where(np.power(self.adj_matrix, i) > 0)
            for x, y in zip(path[0], path[1]):
                if x != y:
                    if path_lengths[x, y] == 0:
                        path_lengths[x, y] = 1
                        path_lengths[y, x] = 1
        self.m = self.n * (self.n - 1)
        avg_path_length = np.sum(path_lengths) / (self.m)
        path_diameter = np.max(path_lengths)
        return (avg_path_length, path_diameter)

    def step(self):
        for i in range(self.nr_steps):
            (row, col) = (np.zeros(self.n), np.zeros(self.n + 1))
            adj_prob = np.divide(np.sum(self.adj_matrix, axis=0), np.sum(self.adj_matrix))
            possible_choices = np.where(adj_prob)
            index = np.random.choice(possible_choices[0].tolist(), self.c, p=adj_prob.tolist())
            for i in index:
                row[i] = 1
                col[i] = 1
            self.adj_matrix = np.vstack((self.adj_matrix, row))
            self.adj_matrix = np.column_stack((self.adj_matrix, col))
            self.n += 1
        self.m = np.sum(self.adj_matrix[np.where(self.adj_matrix==1)])

def task1():
    erdos = Network(n=100, p=0.1, SELF_EDGES=False)
    erdos.initialize_erdos()
    erdos.build_graph()
    erdos.plot_graph()
    erdos.plot_degree_dist()
    plt.show()

def task2():
    wattStrog = Network(n=20, p=0.1, c=8)
    wattStrog.init_small_world()
    wattStrog.add_shortcuts()
    wattStrog.build_graph()
    wattStrog.plot_graph(CIRCULAR=True)
    plt.show()

def task3():
    alberBasi = Network(n=2, p=0.1, c=8, nr_steps=1000)
    alberBasi.add_edge(0, 1)
    alberBasi.step()
    alberBasi.build_graph()
    alberBasi.plot_graph(STEP=True)
    alberBasi.plot_cum_ddist()
    plt.show()

def task4():
    net = Network()
    net.load_data()
    #net.init_small_world()
    net.build_graph()
    net.c = net.compute_clustering_coefficient()   # Using c as a temporary variable
    net.plot_graph(PATH=True)
    plt.show()

def task5():
    net = Network()
    net.load_data()
    #net.init_small_world()
    net.build_graph()
    (net.c, diameter) = net.compute_path_diameter()   # Using c as a temporary variable
    net.plot_graph(CLUSTER=True)
    plt.show()

if __name__ == "__main__":
    #task1()
    #task2()
    #task3()
    #task4()
    task5()