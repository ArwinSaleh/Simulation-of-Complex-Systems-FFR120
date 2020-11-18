from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import random as rnd
from numpy.core.function_base import linspace, logspace
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

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
    plt.loglog(rel_fire_size2, x_axis2, linestyle='none', marker = 'o', markersize=1, label='0.75 initialized trees')
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

    newX = np.logspace(-4, -1, base=10)
    exponents = np.zeros((7, 2))

    fire_data1 = genfromtxt("N_8_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data1 = genfromtxt("N_8_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree1 = np.divide(fire_data1, tree_data1)
    rel_fire_size1 = sorted(fire_div_tree1, reverse=True)
    x_axis1 = list()
    for i in range(len(rel_fire_size1)):
        x_axis1.append((i+1) / len(rel_fire_size1))
    part_rel_fire1 = list()
    part_axis1 = list()
    for i in range(int(len(rel_fire_size1)/2), int(len(rel_fire_size1))):
        part_rel_fire1.append(rel_fire_size1[i])
        part_axis1.append(x_axis1[i])
    exponents[0], pcov = curve_fit(exp_func, part_rel_fire1, part_axis1)

    fire_data2 = genfromtxt("N_16_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data2 = genfromtxt("N_16_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree2 = np.divide(fire_data2, tree_data2)
    rel_fire_size2 = sorted(fire_div_tree2, reverse=True)
    x_axis2 = list()
    for i in range(len(rel_fire_size2)):
        x_axis2.append((i+1) / len(rel_fire_size2))
    part_rel_fire2 = list()
    part_axis2 = list()
    for i in range(int(len(rel_fire_size2)/2), int(len(rel_fire_size2))):
        part_rel_fire2.append(rel_fire_size2[i])
        part_axis2.append(x_axis2[i])
    exponents[1], pcov = curve_fit(exp_func, part_rel_fire2, part_axis2)

    fire_data3 = genfromtxt("N_32_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data3 = genfromtxt("N_32_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree3 = np.divide(fire_data3, tree_data3)
    rel_fire_size3 = sorted(fire_div_tree3, reverse=True)
    x_axis3 = list()
    for i in range(len(rel_fire_size3)):
        x_axis3.append((i+1) / len(rel_fire_size3))
    part_rel_fire3 = list()
    part_axis3 = list()
    for i in range(int(len(rel_fire_size3)/2), int(len(rel_fire_size3))):
        part_rel_fire3.append(rel_fire_size3[i])
        part_axis3.append(x_axis3[i])
    exponents[2], pcov = curve_fit(exp_func, part_rel_fire3, part_axis3)

    fire_data4 = genfromtxt("N_64_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data4 = genfromtxt("N_64_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree4 = np.divide(fire_data4, tree_data4)
    rel_fire_size4 = sorted(fire_div_tree4, reverse=True)
    x_axis4 = list()
    for i in range(len(rel_fire_size4)):
        x_axis4.append((i+1) / len(rel_fire_size4))
    part_rel_fire4 = list()
    part_axis4 = list()
    for i in range(int(len(rel_fire_size4)/2), int(len(rel_fire_size4))):
        part_rel_fire4.append(rel_fire_size4[i])
        part_axis4.append(x_axis4[i])
    exponents[3], pcov = curve_fit(exp_func, part_rel_fire4, part_axis4)

    fire_data5 = genfromtxt("N_128_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data5 = genfromtxt("N_128_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree5 = np.divide(fire_data5, tree_data5)
    rel_fire_size5 = sorted(fire_div_tree5, reverse=True)
    x_axis5 = list()
    for i in range(len(rel_fire_size5)):
        x_axis5.append((i+1) / len(rel_fire_size5))
    part_rel_fire5 = list()
    part_axis5 = list()
    for i in range(int(len(rel_fire_size5)/2), int(len(rel_fire_size5))):
        part_rel_fire5.append(rel_fire_size5[i])
        part_axis5.append(x_axis5[i])
    exponents[4], pcov = curve_fit(exp_func, part_rel_fire5, part_axis5)
    
    fire_data6 = genfromtxt("N_256_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data6 = genfromtxt("N_256_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree6 = np.divide(fire_data6, tree_data6)
    rel_fire_size6 = sorted(fire_div_tree6, reverse=True)
    x_axis6 = list()
    for i in range(len(rel_fire_size6)):
        x_axis6.append((i+1) / len(rel_fire_size6))
    part_rel_fire6 = list()
    part_axis6 = list()
    for i in range(int(len(rel_fire_size6)/2), int(len(rel_fire_size6))):
        part_rel_fire6.append(rel_fire_size6[i])
        part_axis6.append(x_axis6[i])
    exponents[5], pcov = curve_fit(exp_func, part_rel_fire6, part_axis6)
    
    fire_data7 = genfromtxt("N_512_fires_p0.001_f0.05.csv", delimiter=",")
    tree_data7 = genfromtxt("N_512_trees_p0.001_f0.05.csv", delimiter=",")
    fire_div_tree7 = np.divide(fire_data7, tree_data7)
    rel_fire_size7 = sorted(fire_div_tree7, reverse=True)
    x_axis7 = list()
    for i in range(len(rel_fire_size7)):
        x_axis7.append((i+1) / len(rel_fire_size7))
    part_rel_fire7 = list()
    part_axis7 = list()
    for i in range(int(len(rel_fire_size7)/2), int(len(rel_fire_size7))):
        part_rel_fire7.append(rel_fire_size7[i])
        part_axis7.append(x_axis7[i])
    exponents[6], pcov = curve_fit(exp_func, part_rel_fire7, part_axis7)

    tau = np.zeros((len(exponents), 1))
    for i in range(len(exponents)):
        tau[i] = 1 - exponents[i, 1]
    
    inverted_N = [1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512]

    print(tau)

    plt.figure()
    plt.xlabel('Relative fire size')
    plt.ylabel('cCDF')
    plt.loglog(rel_fire_size1, x_axis1, linestyle='none', marker='o', markersize=1, label='N = 8')
    plt.loglog(rel_fire_size2, x_axis2, linestyle='none', marker = 'o', markersize=1, label='N = 16')
    plt.loglog(rel_fire_size3, x_axis3, linestyle='none', marker = 'o', markersize=1, label='N = 32')
    plt.loglog(rel_fire_size4, x_axis4, linestyle='none', marker = 'o', markersize=1, label='N = 64')
    plt.loglog(rel_fire_size5, x_axis5, linestyle='none', marker = 'o', markersize=1, label='N = 128')
    plt.loglog(rel_fire_size6, x_axis6, linestyle='none', marker = 'o', markersize=1, label='N = 256')
    plt.loglog(rel_fire_size7, x_axis7, linestyle='none', marker = 'o', markersize=1, label='N = 512')
    plt.legend()
    plt.xlim(10**(-5), 10**(1))
    plt.title('p = 0.001    f = 0.05        p/f = 0.02')
    plt.figure()
    plt.plot(exponents[:, 1], inverted_N, linestyle='none', marker='o', markersize=4, label='Exponents as a function of 1/N')
    z = Polynomial.fit(exponents[:, 1], inverted_N, 1)
    plt.plot(*z.linspace(), label='Linear regression')
    xs = np.linspace(-0.7, 0)
    ys = np.zeros((len(xs), 1))
    plt.plot(xs, ys, linestyle='--', color='red')
    plt.legend()
    plt.show()

task2()
#task3_1()
#task3_2()
#task4()
