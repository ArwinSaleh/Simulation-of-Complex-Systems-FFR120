from math import log
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from numpy.lib.function_base import append
import math
from numpy import savetxt
from numpy import asarray
import pandas as pd 
from numpy import genfromtxt

class BrownianParticles:
    def __init__(self, nr_particles, grid_length, nr_steps, D_T, D_R, tau, DRAW=True, T0=0, r_c=5, DRAW_PATH=False):
        self.nr_particles = nr_particles
        self.nr_steps = nr_steps
        self.grid_length = grid_length
        self.active_particles = np.ones((nr_particles, 1))  
        self.X = np.random.uniform(0, grid_length, size=(nr_particles, 1))
        self.Y = np.random.uniform(0, grid_length, size=(nr_particles, 1))
        self.velocities = np.zeros((nr_particles, 2))
        self.particle_historyX = list()
        self.particle_historyY = list()
        self.historyX = [[] for i in range(0, nr_particles)]
        self.historyY = [[] for i in range(0, nr_particles)]
        self.historyMSD = [[] for i in range(0, nr_particles)]
        self.D_T = D_T
        self.D_R = D_R
        self.T0 = T0
        self.r_c=r_c
        self.theta = np.random.uniform(-1, 1, size=(nr_particles, 1))
        self.v = np.random.uniform(-0.000001, 0.000001, size=(nr_particles, 1))
        self.tau = tau
        self.time_step = 1
        self.DRAW = DRAW
        self.DRAW_PATH = DRAW_PATH

    def draw_particles(self, LEGEND):
        plt.clf()
        
        if self.DRAW_PATH:
            for i in range(self.nr_particles):
                    plt.plot(self.historyX[i], self.historyY[i], '.', markevery=[len(self.particle_historyX)-1], label='v = ' + str(self.v[i]) + " \u03BCm/s")
        else:
            plt.plot(self.X, self.Y, '.', markersize=10)

        if LEGEND:
            plt.legend()
        plt.axis([0, self.grid_length, 0, self.grid_length])
        plt.title('D_T = ' + str(self.D_T) + 
                    "      D_R = " + str(self.D_R) + 
                    "     Number of Particles = " + 
                    str(self.nr_particles) + "\nt = " + 
                    str(self.time_step))

        plt.xlabel('x [\u03BCm]')
        plt.ylabel('y [\u03BCm]')
        plt.draw()
        plt.pause(0.0000001)

    def update_velocity(self, i):
        Wx = np.random.normal(0, 1)
        Wy = np.random.normal(0, 1)
        self.velocities[i, 0] = self.v[i] * math.cos(self.theta[i]) + Wx * math.sqrt(2 * self.D_T)
        self.velocities[i, 1] = self.v[i] * math.sin(self.theta[i]) + Wy * math.sqrt(2 * self.D_T)
    
    def update_theta(self, i):
        Wo = np.random.normal(0, 1)
        self.theta[i] = self.theta[i] + Wo * math.sqrt(2 * self.D_R)
    
    def update_position(self, i):

        self.X[i] += self.velocities[i, 0]
        self.Y[i] += self.velocities[i, 1]
        if self.X[i] > self.grid_length:
            self.X[i] -= self.velocities[i, 0]
        if self.Y[i] > self.grid_length:
            self.Y[i] -= self.velocities[i, 1]
        if self.X[i] < 0:
            self.X[i] -= self.velocities[i, 0]
        if self.Y[i] < 0:
            self.Y[i] -= self.velocities[i, 1]
        if self.collision(i):
            self.X[i] -= self.velocities[i, 0]
            self.Y[i] -= self.velocities[i, 1]
        self.historyX[i].append(float(self.X[i]))
        self.historyY[i].append(float(self.Y[i]))

    def compute_MSD(self, i, step):
        MSD = 0
        for t in range(step):
            if t + self.tau < len(self.historyX[0]):
                MSD += ((self.historyX[i][t + self.tau] - self.historyX[i][t])**2 + 
                                (self.historyY[i][t + self.tau] - self.historyY[i][t])**2)
        MSD /= (self.nr_steps - 1 * (self.tau - 1)) 
        self.historyMSD[i].append(MSD)
    
    def compute_torque(self, n):
        inactive_particles = np.where(self.active_particles == 0)[0]
        active_X = np.delete(self.X, inactive_particles)
        active_Y = np.delete(self.Y, inactive_particles)
        active_X = np.delete(self.X, 0)
        active_Y = np.delete(self.Y, 0)
        
        v_nx = self.velocities[n, 0]
        v_ny = self.velocities[n, 1]
        '''
        v_n = np.array([[v_nx, v_ny],]*(self.nr_particles-1)).transpose()
        v_n = v_n / np.linalg.norm(v_n)
        v_n_r_ni = v_n *   
        r_ni_2 = np.power(np.add(np.abs(np.subtract(self.X[n], active_X)), np.abs(np.subtract(self.Y[n], active_Y))), 2)
        cross = np.subtract(v_nx * (np.abs(np.subtract(self.Y[n], active_Y))), v_ny * np.abs(np.subtract(self.X[n], active_X)))
        # Cross = Vx * (Yn - Yi) - Vy * (Xn - Xi)
        T_n = np.multiply(self.T0, np.dot(np.divide(v_n_r_ni, r_ni_2), cross))
        self.theta[n] += T_n
        '''
        v_n = np.array([[v_nx, v_ny],]*(self.nr_particles-1)).transpose()
        r_ni_2 = np.add(np.power(np.subtract(self.X[n], active_X), 2), np.power(np.subtract(self.Y[n], active_Y), 2))
        v_n_hat = v_n / np.linalg.norm(v_n)
        r_ni = np.array([np.subtract(self.X[n], active_X), np.subtract(self.Y[n], active_Y)])
        r_ni_hat = r_ni / np.linalg.norm(r_ni)
        cross = np.subtract(v_n_hat[0] * (np.abs(np.subtract(self.Y[n], active_Y))), v_n_hat[1] * np.abs(np.subtract(self.X[n], active_X)))
        div = np.divide(np.dot(np.transpose(v_n_hat), r_ni_hat), r_ni_2)
        div[np.isnan(div)] = 0
        T_n = np.multiply(self.T0, np.dot(div, cross))
        self.theta[n] += np.sum(T_n)
            
    def step(self, TAU=1, SAVEFIG=False, SAVETHRESH=50, LEGEND=True, MSD=False):
        for step in range(self.nr_steps):
            if SAVEFIG and self.time_step == SAVETHRESH+1:
                plt.savefig('particles_at_50')
            if (self.DRAW):
                self.draw_particles(LEGEND)
            else:
                print("t = " + str(self.time_step))
            for i in range(self.nr_particles):
                self.update_theta(i)
                if self.time_step > 1:
                    self.compute_torque(i)
                self.update_velocity(i)
                self.update_position(i)
                if MSD:
                    self.compute_MSD(i, step)
            self.time_step += 1

    def collision(self, i):
        pop_on_X = np.where(self.X == self.X[i])[0]
        pop_on_Y = np.where(self.Y == self.Y[i])[0]
        pop_on_site = np.union1d(pop_on_X, pop_on_Y)
        if len(pop_on_site) > 1:
            return True
        return False
    
    def clear_all(self):
        self.historyX = [[] for i in range(0, self.nr_particles)]
        self.historyY = [[] for i in range(0, self.nr_particles)]
        self.historyMSD = [[] for i in range(0, self.nr_particles)]
        self.time_step = 0
    
    def plot_many_iterations(self):
        runs = 100
        run = genfromtxt("HW3/data/run_1.csv", delimiter=",")
        for i in range(1, runs):
            run = np.add(genfromtxt("HW3/data/run_" + str(i) + ".csv", delimiter=","), genfromtxt("HW3/data/run_" + str(i+1) + ".csv", delimiter=","))

        run = np.divide(run, runs)

        t = np.zeros((len(run[0]), 1))
        for i in range(len(t)):
                t[i] = i

        for i in range(len(run[:, 0])):
            plt.loglog(t, run[i], label='v = ' + str(self.v[i]) + " \u03BCm/s")

        plt.title("Averaged over " + str(runs) + " runs" + '\nD_T = ' + str(self.D_T) + 
                    "      D_R = " + str(self.D_R) + 
                    "     Number of Particles = " + 
                    str(self.nr_particles))
        
        plt.xlabel('t')
        plt.ylabel('MSD [\u03BCm^2]')
        plt.legend()
        plt.axis([2, 250, 0, 1])
        plt.show()

def plot_iterations(SAVE_DATA=True):
    brown = BrownianParticles(nr_particles=4, grid_length=200, nr_steps=200, D_T=0.25, D_R=0.15, tau=1, DRAW=False)
    brown.v=[0, 1, 2, 3]
    run = 1
    runs = 100
    if SAVE_DATA:
        for i in range(runs):
            brown.step(SAVEFIG=False)
            pd.DataFrame(asarray(brown.historyMSD)).to_csv("HW3/data/run_" + str(run) + ".csv", header=None)
            brown.clear_all()
            run += 1
    
    brown.plot_many_iterations()

def task1():
    brown = BrownianParticles(nr_particles=4, grid_length=1000, nr_steps=100, D_T=0.25, D_R=0.15, tau=1, DRAW=True)
    brown.v = [0, 1, 2, 3]
    brown.step(SAVEFIG=True, SAVETHRESH=50)
    t = np.zeros((brown.nr_steps, 1))
    for i in range(brown.nr_steps):
        t[i] = i

def task2():
    brown = BrownianParticles(nr_particles=100, grid_length=100, nr_steps=10000, D_T=0.0052, D_R=0.0002, tau=1, DRAW=True, r_c=100, T0=1)
    brown.step(SAVEFIG=False, SAVETHRESH=50, LEGEND=False)

    '''
    t = np.zeros((brown.nr_steps, 1))
    for i in range(brown.nr_steps):
        t[i] = i
    plt.figure()
    for i in range(0, brown.nr_particles):
        plt.loglog(t, brown.historyMSD[i], label='v = ' + str(brown.v[i]) + " \u03BCm/s")
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('MSD [\u03BCm^2]')
    plt.title('D_T = ' + str(brown.D_T) + 
                    "      D_R = " + str(brown.D_R) + 
                    "     Number of Particles = " + 
                    str(brown.nr_particles))
    plt.show()
    '''

#save_data()
#task1()
#plot_iterations(SAVE_DATA=False)
task2()