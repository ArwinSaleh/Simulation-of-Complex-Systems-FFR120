from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt


run = genfromtxt("run_1.csv", delimiter=",")
for i in range(1, 10):
    run = np.add(genfromtxt("run_" + str(i) + ".csv", delimiter=","), genfromtxt("run_" + str(i+1) + ".csv", delimiter=","))

run = np.divide(run, 10)

t = np.zeros((len(run[0]), 1))
for i in range(len(t)):
        t[i] = i

for i in range(len(run[:, 0])):
    plt.loglog(t, run[i])

plt.show()