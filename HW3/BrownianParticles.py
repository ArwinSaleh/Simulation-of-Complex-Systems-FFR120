import numpy as np
import matplotlib.pyplot as plt

class BrownianParticles:
    def __init__(self, nr_particles, grid_length):
        self.nr_particles = nr_particles
        self.grid_length = grid_length
        self.particles = np.zeros((self.nr_particles, 1))
        self.velocities = np.zeros((self.nr_particles, 2))
        