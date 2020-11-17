import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import random as rnd
import math
from numpy.core.function_base import linspace
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def task2():
    fire_data = genfromtxt("fires_p0.01_f0.1.csv", delimiter=",")
    tree_data = genfromtxt("trees_p0.01_f0.1.csv", delimiter=",")

    fire_div_tree = np.divide(fire_data, tree_data)

    rel_fire_size = sorted(fire_div_tree, reverse=True)

    x_axis = list()

    for i in range(len(fire_data)):
        x_axis.append((i+1) / len(fire_data))

    half_rel_fire = list()
    half_axis = list()
    for i in range(int(len(fire_data)/4)):
        half_rel_fire.append(rel_fire_size[i])
        half_axis.append(x_axis[i])

    plt.figure()
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(rel_fire_size, x_axis, linestyle='none', marker='o', markersize=1)
    x = linspace(10**(-3), 10, 1000)
    newX = np.logspace(-3, 1, base=10)
    popt, pcov = curve_fit(exp_func, half_axis, half_rel_fire)
    plt.plot(newX, exp_func(newX*128, *popt), 'r-', 
         label="({0:.3f}*x**{1:.3f})".format(*popt))
    print("Exponential Fit: y = (a*(x**b))")
    print("\ta = popt[0] = {0}\n\tb = popt[1] = {1}".format(*popt))
    plt.show()

def task3():
    fire_data = genfromtxt("fires_p0.01_f0.1.csv", delimiter=",")
    tree_data = genfromtxt("trees_p0.01_f0.1.csv", delimiter=",")

    fire_div_tree = np.divide(fire_data, tree_data)

    rel_fire_size = sorted(fire_div_tree, reverse=True)

    x_axis = list()

    for i in range(len(rel_fire_size)):
        x_axis.append((i+1) / len(rel_fire_size))

    X = list()
    tau = 1.15
    xmin = min(rel_fire_size)
    for i in range(len(rel_fire_size)):
        r_i = rnd.uniform(0, 1)
        X.append(xmin*(1-r_i)**(-1.0/(tau - 1.0)))

    X = sorted(X, reverse=True)

    plt.figure()
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(X, x_axis, linestyle='none', marker='o', markersize=1, label='Synthetic')
    plt.loglog(rel_fire_size, x_axis, linestyle='none', marker = 'o', markersize=1, label='Simulated')
    plt.legend()
    plt.show()

def exp_func(x, a, b):
    return a * np.power(x, b)

task2()
#task3()