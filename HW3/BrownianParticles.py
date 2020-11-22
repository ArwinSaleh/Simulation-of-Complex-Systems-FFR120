import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append

class BrownianParticles:
    def __init__(self, nr_particles, grid_length, nr_steps, D_T, D_R, v):
        self.nr_particles = nr_particles
        self.nr_steps = nr_steps
        self.grid_length = grid_length
        self.particles = np.ones((nr_particles, 1))
        self.X = np.zeros((nr_particles, 1))
        self.Y = np.zeros((nr_particles, 1))
        self.velocities = np.zeros((nr_particles, 2))
        self.particle_historyX = list()
        self.particle_historyY = list()
        self.particle_historyX.append(0)
        self.particle_historyY.append(0)
        self.D_T = D_T
        self.D_R = D_R
        self.theta = np.zeros((nr_particles, 1))
        self.v = v

    def draw_particles(self):
        plt.clf()
        plt.plot(self.particle_historyX, self.particle_historyY, "r-")
        plt.axis([-self.grid_length, self.grid_length, -self.grid_length, self.grid_length])
        plt.draw()
        plt.pause(0.0001)


    def update_velocity(self, i):
        Wx = np.random.normal(0, 1)
        Wy = np.random.normal(0, 1)
        self.velocities[i, 0] += Wx * np.sqrt(2 * self.D_T)
        self.velocities[i, 1] += Wy * np.sqrt(2 * self.D_T)
    
    def update_theta(self, i):
        Wo = np.random.normal(0, 1)
        self.theta
    
    def update_position(self, i):
        self.X[i] += self.velocities[i, 0]
        self.Y[i] += self.velocities[i, 1]
        self.particle_historyX.append(self.X[i])
        self.particle_historyY.append(self.Y[i])

    def step(self):
        for step in range(self.nr_steps):
            self.draw_particles()
            for i in range(self.nr_particles):
                self.update_velocity(i)
                self.update_position(i)

brown = BrownianParticles(nr_particles=1, grid_length=100, nr_steps=1000, D_T=0.2, D_R=0.5, v=0.5)
brown.step()