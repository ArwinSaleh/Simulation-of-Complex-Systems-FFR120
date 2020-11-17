import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
from numpy import genfromtxt
import random as rnd
import math

fire_data = genfromtxt("fires_p0.01_f0.1.csv", delimiter=",")
tree_data = genfromtxt("trees_p0.01_f0.1.csv", delimiter=",")

fire_div_tree = np.divide(fire_data, tree_data)

rel_fire_size = sorted(fire_div_tree, reverse=True)

x_axis = list()

for i in range(len(fire_data)):

    x_axis.append((i+1) / len(fire_data))

X = list()
tau = 1.15
xmin = min(fire_data)/(128*128)
for i in range(len(fire_data)):
    r_i = rnd.uniform(0, 1)
    X.append(xmin*(1-r_i)**(-1.0/(tau - 1)))

X = sorted(X, reverse=True)

plt.figure()
plt.xlabel('Relative fire size')
plt.ylabel('cCDF')
plt.loglog(X, x_axis, linestyle='none', marker='o', markersize=1, label='Synthetic')
plt.loglog(rel_fire_size, x_axis, linestyle='none', marker = 'o', markersize=1, label='Simulated')
plt.legend()
plt.show()