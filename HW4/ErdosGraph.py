import numpy as np
from scipy.sparse import csr_matrix
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sb

class Network:
    def __init__(self, n=0, c=0, p=0, SELF_EDGES=False):
        self.G = nx.Graph()
        self.p = p
        self.c = c
        self.n = n              # Nodes
        self.m = n * (n - 1)    # Edges
        self.adj_matrix = np.zeros((self.n, self.n))
        self.SELF_EDGES = SELF_EDGES

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
                    self.adj_matrix[i, i + j] = 1
                    self.adj_matrix[i + j, i] = 1
                else:
                    self.adj_matrix[i, i + j] = 1
                    self.adj_matrix[i + j, i] = 1
                    
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

    def build_graph(self):
        v1 = np.where(self.adj_matrix == 1)[0]
        v2 = np.where(self.adj_matrix == 1)[1]
        
        for i in range(len(v1)):
            self.G.add_edge(v1[i], v2[i])

    def plot_graph(self, CIRCULAR=False):
        plt.figure()
        
        if CIRCULAR:
            plt.title('p = ' + str(self.p) + '      c = ' + str(self.c) + '      n = ' + str(self.n) + '     m = ' + str(self.m))
            nx.draw_circular(self.G)
        else:
            plt.title('p = ' + str(self.p) + '      n = ' + str(self.n) + '     m = ' + str(self.m))
            nx.draw(self.G)
    
    def plot_degree_dist(self):
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        plt.figure()
        plt.xlabel('Degrees')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution' + '\np = ' + str(self.p) + '      n = ' + str(self.n))
        sb.distplot(sorted(degrees, reverse=True), kde=True, bins=20)
        
def task1():
    erdos = Network(n=100, p=0.1, SELF_EDGES=False)
    erdos.initialize_erdos()
    erdos.build_graph()
    erdos.plot_graph()
    erdos.plot_degree_dist()
    plt.show()

def task2():
    wattStrog = Network(n=20, p=0.1, c=5)
    wattStrog.init_small_world()
    wattStrog.add_shortcuts()
    wattStrog.build_graph()
    wattStrog.plot_graph(CIRCULAR=True)
    plt.show()

if __name__ == "__main__":
    #task1()
    #task2()