import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import random as rnd
from scipy.optimize import curve_fit

def exp_func(x, a, b):
    return a * np.power(x, b)

def task2():
    fire_data1 = genfromtxt("fires_p0.001_f0.05_NOT_INIT.csv", delimiter=",")
    tree_data1 = genfromtxt("trees_p0.001_f0.05_NOT_INIT.csv", delimiter=",")

    fire_div_tree1 = np.divide(fire_data1, tree_data1)

    rel_fire_size1 = sorted(fire_div_tree1, reverse=True)

    fire_data2 = genfromtxt("fires_p0.001_f0.05_INIT.csv", delimiter=",")
    tree_data2 = genfromtxt("trees_p0.001_f0.05_INIT.csv", delimiter=",")

    fire_div_tree2 = np.divide(fire_data2, tree_data2)

    rel_fire_size2 = sorted(fire_div_tree2, reverse=True)

    x_axis1 = list()
    for i in range(len(rel_fire_size1)):
        x_axis1.append((i+1) / len(rel_fire_size1))

    x_axis2 = list()
    for i in range(len(rel_fire_size2)):
        x_axis2.append((i+1) / len(rel_fire_size2))


    plt.figure()
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(rel_fire_size1, x_axis1, linestyle='none', marker='o', markersize=1, label='NO initialized trees')
    plt.loglog(rel_fire_size2, x_axis2, linestyle='none', marker = 'o', markersize=1, label='0.25 initialized trees')
    plt.xlim(10**(-4), 10**(1))
    plt.title('p = 0.001    f = 0.05        p/f = 0.02')
    plt.legend()
    plt.show()

def task3_1():
    fire_data = genfromtxt("fires_p0.001_f0.05.csv", delimiter=",")
    tree_data = genfromtxt("trees_p0.001_f0.05.csv", delimiter=",")

    fire_div_tree = np.divide(fire_data, tree_data)

    rel_fire_size = sorted(fire_div_tree, reverse=True)

    x_axis = list()

    for i in range(len(fire_data)):
        x_axis.append((i+1) / len(fire_data))

    part_rel_fire = list()
    part_axis = list()
    for i in range(int(len(rel_fire_size)/4), int(len(rel_fire_size))):
        part_rel_fire.append(rel_fire_size[i])
        part_axis.append(x_axis[i])

    plt.figure()
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(rel_fire_size, x_axis, linestyle='none', marker='o', markersize=1)
    newX = np.logspace(-4, -1, base=10)
    popt, pcov = curve_fit(exp_func, part_rel_fire, part_axis)
    plt.loglog(newX, exp_func(newX, *popt), 'r-', 
         label="({0:.3f} * x ** {1:.3f})".format(*popt) + '\n   tau = ' + str(round(1 - popt[1], 3)))
    plt.title('p = 0.001    f = 0.05        p/f = 0.02')
    plt.legend()
    plt.show()

def task3_2():
    fire_data = genfromtxt("fires_p0.001_f0.05.csv", delimiter=",")
    tree_data = genfromtxt("trees_p0.001_f0.05.csv", delimiter=",")

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
    plt.loglog(X, x_axis, linestyle='none', marker='o', markersize=1, label='Synthetic  (tau = 1.150)', color='orange')
    plt.loglog(rel_fire_size, x_axis, linestyle='none', marker = 'o', markersize=1, label='Simulated (tau = 1.236)')
    plt.xlim(10**(-4), 10**(1))
    plt.title('p = 0.001    f = 0.05        p/f = 0.02')
    plt.legend()
    plt.show()

def task4():
    fire_data1 = genfromtxt("N_256_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data1 = genfromtxt("N_256_trees_p0.001_f0.05.csv", delimiter=",")

    fire_div_tree1 = np.divide(fire_data1, tree_data1)

    rel_fire_size1 = sorted(fire_div_tree1, reverse=True)

    fire_data2 = genfromtxt("N_512_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data2 = genfromtxt("N_512_trees_p0.001_f0.05.csv", delimiter=",")

    fire_div_tree2 = np.divide(fire_data2, tree_data2)

    rel_fire_size2 = sorted(fire_div_tree2, reverse=True)

    x_axis1 = list()
    for i in range(len(rel_fire_size1)):
        x_axis1.append((i+1) / len(rel_fire_size1))

    x_axis2 = list()
    for i in range(len(rel_fire_size2)):
        x_axis2.append((i+1) / len(rel_fire_size2))

    part_rel_fire1 = list()
    part_axis1 = list()
    for i in range(int(len(rel_fire_size1)/4), int(len(rel_fire_size1))):
        part_rel_fire1.append(rel_fire_size1[i])
        part_axis1.append(x_axis1[i])

    part_rel_fire2 = list()
    part_axis2 = list()
    for i in range(int(len(rel_fire_size2)/4), int(len(rel_fire_size2))):
        part_rel_fire2.append(rel_fire_size2[i])
        part_axis2.append(x_axis2[i])

    plt.figure()
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(rel_fire_size1, x_axis1, linestyle='none', marker='o', markersize=1, label='N = 256')
    plt.loglog(rel_fire_size2, x_axis2, linestyle='none', marker = 'o', markersize=1, label='N = 512')
    newX = np.logspace(-4, -1, base=10)
    popt, pcov = curve_fit(exp_func, part_rel_fire1, part_axis1)
    plt.loglog(newX, exp_func(newX, *popt), 'r-', 
         label="({0:.3f} * x ** {1:.3f})".format(*popt) + '\n   tau = ' + str(round(1 - popt[1], 3)), color='red')
    popt2, pcov = curve_fit(exp_func, part_rel_fire2, part_axis2)
    plt.loglog(newX, exp_func(newX, *popt2), 'r-', 
         label="({0:.3f} * x ** {1:.3f})".format(*popt2) + '\n   tau = ' + str(round(1 - popt2[1], 3)), color='green')
    plt.xlim(10**(-4), 10**(1))
    plt.title('p = 0.001    f = 0.05        p/f = 0.02')
    plt.legend()
    plt.show()


#task2()
#task3_1()
#task3_2()
task4()
