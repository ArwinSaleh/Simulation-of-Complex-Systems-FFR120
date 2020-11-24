from os import path
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
from numpy import asarray




'''    
runs = 100

run = genfromtxt("HW3/data/run_1.csv", delimiter=",")
for i in range(1, runs):
    run = np.add(genfromtxt("HW3/data/run_" + str(i) + ".csv", delimiter=","), genfromtxt("HW3/data/run_" + str(i+1) + ".csv", delimiter=","))

run = np.divide(run, runs)

t = np.zeros((len(run[0]), 1))
for i in range(len(t)):
        t[i] = i

for i in range(len(run[:, 0])):
    plt.loglog(t, run[i])

plt.show()
'''
