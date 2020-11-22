import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append
import math

class BrownianParticles:
    def __init__(self, nr_particles, grid_length, nr_steps, D_T, D_R, v):
        self.nr_particles = nr_particles
        self.nr_steps = nr_steps
        self.grid_length = grid_length
        self.particles = np.ones((nr_particles, 1))
        self.X = np.random.uniform(-grid_length, grid_length, size=(nr_particles, 1))
        self.Y = np.random.uniform(-grid_length, grid_length, size=(nr_particles, 1))
        self.velocities = np.zeros((nr_particles, 2))
        self.particle_historyX = list()
        self.particle_historyY = list()
        self.historyX = [[] for i in range(0, nr_particles)]
        self.historyY = [[] for i in range(0, nr_particles)]
        self.D_T = D_T
        self.D_R = D_R
        self.theta = np.ones((nr_particles, 1))
        self.v = v

    def draw_particles(self):
        plt.clf()
        for i in range(self.nr_particles):
            if len(self.historyX) > 1:
                plt.plot(self.historyX[i], self.historyY[i], '-o', markevery=[len(self.particle_historyX)-1])
        plt.axis('equal')
        plt.draw()
        plt.pause(0.00000001)

    def update_velocity(self, i):
        Wx = np.random.normal(0, 1)
        Wy = np.random.normal(0, 1)
        self.velocities[i, 0] = self.v * math.cos(self.theta[i]) + Wx * math.sqrt(2 * self.D_T)
        self.velocities[i, 1] = self.v * math.sin(self.theta[i]) + Wy * math.sqrt(2 * self.D_T)
    
    def update_theta(self, i):
        Wo = np.random.normal(0, 1)
        self.theta[i] = -Wo * math.sqrt(2 * self.D_R)
    
    def update_position(self, i):
        self.X[i] += self.velocities[i, 0]
        self.Y[i] += self.velocities[i, 1]

    def step(self):
        for step in range(self.nr_steps):
            self.draw_particles()
            for i in range(self.nr_particles):
                self.update_theta(i)
                self.update_velocity(i)
                self.update_position(i)
                self.historyX[i].append(float(self.X[i]))
                self.historyY[i].append(float(self.Y[i]))

def main():
    brown = BrownianParticles(nr_particles=20, grid_length=50, nr_steps=1000, D_T=0.5, D_R=0.5, v=0.1)
    brown.step()

main()