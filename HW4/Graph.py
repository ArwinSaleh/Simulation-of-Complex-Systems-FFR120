import matplotlib.pyplot as plt

class Graph(object):
    def __init__(self, size):
        self.adjacency_matrix = []
        for i in range(size):
            self.adjacency_matrix.append([0 for i in range(size)])
        self.size = size
    
    def add_edge(self, v1, v2):
        if v1 == v2:
            print("Same vertices!")
        self.adjacency_matrix[v1][v2] = 1
        self.adjacency_matrix[v2][v1] = 1

    def del_edge(self, v1, v2):
        if self.adjacency_matrix[v1][v2] == 0:
            print("There is no edge " + str(v1) + "-" + str(v2) + " in the graph!")
            return
        self.adjacency_matrix[v1][v2] = 0
        self.adjacency_matrix[v2][v1] = 0
    
    def get_graph_size(self):
        return self.size
    
    def show_graph(self):
        for row in self.adjacency_matrix:
            for val in row:
                print('{:4}'.format(val)),
            print

def main():
    g = Graph(5)

    g.add_edge(1, 2)
    g.add_edge(3, 4)

    g.show_graph()

if __name__ == "__main__":
    main()