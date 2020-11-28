import numpy as np
from scipy.sparse import csr_matrix
import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sb

class WattsStrogatz:
    def __init__(self, n, c, p):
        self.n = n
        self.c = c
        self.p = 0
        
