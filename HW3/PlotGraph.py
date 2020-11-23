from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

tau1 = genfromtxt("tau = 1.csv", delimiter=",")
tau2 = genfromtxt("tau = 1.csv", delimiter=",")
tau3 = genfromtxt("tau = 1.csv", delimiter=",")
tau4 = genfromtxt("tau = 1.csv", delimiter=",")
tau5 = genfromtxt("tau = 1.csv", delimiter=",")

MSD = list()

tau1avg = np.sum(tau1[0, :]) / len(tau1[0])
tau2avg = np.sum(tau1[0, :]) / len(tau2[0])
tau3avg = np.sum(tau1[0, :]) / len(tau3[0])
tau4avg = np.sum(tau1[0, :]) / len(tau4[0])
tau5avg = np.sum(tau1[0, :]) / len(tau5[0])

plt.loglog([1, 2, 3, 4, 5], [tau1avg, tau2avg, tau3avg, tau4avg, tau5avg])
plt.show()