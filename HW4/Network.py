import numpy as np
from scipy.sparse import csr_matrix
import math
import networkx as nx

class Network:
    def __init__(self, n, p, SELF_EDGES=False):
        self.n = n              # Nodes
        self.m = n * (n - 1)    # Edges
        self.adj_matrix = np.zeros((n, n))
        if SELF_EDGES:
            np.fill_diagonal(self.adj_matrix, 2)

    def add_edge(self, v1, v2):
        if v1 != v2:
            self.adj_matrix[v1, v2] = 1
            self.adj_matrix[v2, v1] = 1
        
    def del_edge(self, v1, v2):
        self.adj_matrix[v1, v2] = 0
        self.adj_matrix[v2, v1] = 0

    def initialize_erdos(self):
        self.adj_matrix = np.ones((self.n, self.n))
        r = np.random.uniform(0, 1)


def main():
    net = Network(10)
    net.initialize_erdos()
    math.perm()

if __name__ == "__main__":
    main()    