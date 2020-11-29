import numpy as np
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import math
import scipy
import seaborn as sns
from scipy.stats import binom, powerlaw,norm

sns.set()
number_nodes = 100
probability_edge = 0.25
adjacency_matrix = np.zeros((number_nodes, number_nodes))
index = list(itertools.combinations(range(number_nodes),2))

for i in index:
    random_number = np.random.random(1)
    if random_number < probability_edge:
        adjacency_matrix[i[0]][i[1]] = 1
        adjacency_matrix[i[1]][i[0]] = 1

rows, cols = np.where(adjacency_matrix == 1)    
edges = zip(rows.tolist(), cols.tolist())        

gr = nx.Graph()
gr.add_edges_from(edges)
plt.figure(3,figsize=(12,12)) 
nx.draw_random(gr, node_size=200, font_size=10)
plt.show()


degrees = [gr.degree(n) for n in gr.nodes()]
# The theoretical binomial distribution plot together with the actual distribution of degrees in the nertwork
n = number_nodes
p = probability_edge
data_binom = binom.rvs(n,p,loc=0,size=number_nodes)

ax=sns.distplot(data_binom, kde=True, bins=20,color='teal', label ='Theoretical')
ax=sns.distplot(degrees,kde=True, bins=20, color = 'red',label = 'Degrees')#,hist_kws={"linewidth": 25,'alpha':1})
ax.set(xlabel='Degrees', ylabel='Frequency')
plt.legend()

plt.show()