from math import log
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
import math
from numpy import savetxt
from numpy import asarray

class BrownianParticles:
    def __init__(self, nr_particles, grid_length, nr_steps, D_T, D_R, v, tau, DRAW=True):
        self.nr_particles = nr_particles
        self.nr_steps = nr_steps
        self.grid_length = grid_length
        self.particles = np.ones((nr_particles, 1))
        self.X = np.random.uniform(-grid_length/5, grid_length/5, size=(nr_particles, 1))
        self.Y = np.random.uniform(-grid_length/5, grid_length/5, size=(nr_particles, 1))
        self.velocities = np.zeros((nr_particles, 2))
        self.particle_historyX = list()
        self.particle_historyY = list()
        self.historyX = [[] for i in range(0, nr_particles)]
        self.historyY = [[] for i in range(0, nr_particles)]
        self.historyMSD = [[] for i in range(0, nr_particles)]
        self.D_T = D_T
        self.D_R = D_R
        self.theta = np.ones((nr_particles, 1))
        self.v = v
        self.tau = tau
        self.time_step = 1
        self.DRAW = DRAW

    def draw_particles(self):
        plt.clf()
        plt.title('t = ' + str(self.time_step))
        for i in range(self.nr_particles):
            if len(self.historyX) > 1:
                plt.plot(self.historyX[i], self.historyY[i], '-o', markevery=[len(self.particle_historyX)-1], label='v = ' + str(self.v[i]))
        plt.axis([-2 - self.grid_length, self.grid_length + 2, -2 - self.grid_length, self.grid_length + 2])
        plt.legend()
        plt.draw()
        plt.pause(0.00000001)

    def update_velocity(self, i):
        Wx = np.random.normal(0, 1)
        Wy = np.random.normal(0, 1)
        self.velocities[i, 0] = self.v[i] * math.cos(self.theta[i]) + Wx * math.sqrt(2 * self.D_T)
        self.velocities[i, 1] = self.v[i] * math.sin(self.theta[i]) + Wy * math.sqrt(2 * self.D_T)
    
    def update_theta(self, i):
        Wo = np.random.normal(0, 1)
        self.theta[i] = Wo * math.sqrt(2 * self.D_R)
    
    def update_position(self, i):
        self.X[i] += self.velocities[i, 0]
        self.Y[i] += self.velocities[i, 1]
        self.historyX[i].append(int(self.X[i]))
        self.historyY[i].append(float(self.Y[i]))

    def compute_MSD(self, i, step):
        MSD = 0
        for t in range(step):
            if t + self.tau < len(self.historyX[0]):
                MSD += ((self.historyX[i][t + self.tau] - self.historyX[i][t])**2 + 
                                (self.historyY[i][t + self.tau] - self.historyY[i][t])**2)
        MSD /= (self.nr_steps - 1 * (self.tau - 1)) 
        self.historyMSD[i].append(MSD)
            
    def step(self, TAU=1):
        for step in range(self.nr_steps):
            if (self.DRAW):
                self.draw_particles()
            else:
                print("t = " + str(self.time_step))
            for i in range(self.nr_particles):
                self.update_theta(i)
                self.update_velocity(i)
                self.update_position(i)
                self.compute_MSD(i, step)
            self.time_step += 1
    
    def clear_all(self):
        self.historyX = [[] for i in range(0, self.nr_particles)]
        self.historyY = [[] for i in range(0, self.nr_particles)]
        self.historyMSD = [[] for i in range(0, self.nr_particles)]
        self.time_step = 0

def save_data():
    brown = BrownianParticles(nr_particles=4, grid_length=100, nr_steps=1000, D_T=0.02, D_R=0.06, v=[0.00, 0.1, 0.2, 0.3], tau=1, DRAW=False)
    taus = [1, 2, 3, 4, 5]
    for tau in taus:
        brown.step(TAU=tau)
        savetxt("tau = " + str(tau) + ".csv", asarray(brown.historyMSD), delimiter=',')
        brown.clear_all()

    t = np.zeros((brown.nr_steps, 1))
    for i in range(brown.nr_steps):
        t[i] = i**2
    plt.figure()
    for i in range(0, brown.nr_particles):
        plt.loglog(t, brown.historyMSD[i], label='v = ' + str(brown.v[i]))
    plt.legend()
    plt.show()

def main():
    brown = BrownianParticles(nr_particles=4, grid_length=100, nr_steps=1000, D_T=0.22, D_R=0.16, v=[0, 1, 2, 3], tau=1, DRAW=False)
    brown.step()
    t = np.zeros((brown.nr_steps, 1))
    for i in range(brown.nr_steps):
        t[i] = i
    plt.figure()
    for i in range(0, brown.nr_particles):
        plt.loglog(t, brown.historyMSD[i], label='v = ' + str(brown.v[i]))
    plt.legend()
    plt.show()

main()