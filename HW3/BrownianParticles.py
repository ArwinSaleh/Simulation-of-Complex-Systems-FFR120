from math import log
from re import T
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
        self.X_history = list()
        self.Y_history = list()
        self.historyMSD = [[] for i in range(0, nr_particles)]
        self.D_T = D_T
        self.D_R = D_R
        self.T0 = T0
        self.r_c=r_c
        self.theta = np.random.uniform(-1, 1, size=(nr_particles, 1))
        self.v = np.random.uniform(-0.00001, 0.00001, size=(nr_particles, 1))
        self.tau = tau
        self.time_step = 1
        self.DRAW = DRAW
        self.DRAW_PATH = DRAW_PATH

    def initialize_inactive_particles(self):
        self.active_particles[int(0.2 * self.nr_particles) : self.nr_particles] = 0

    def draw_particles(self, LEGEND):
        plt.clf()
        
        if self.DRAW_PATH:
            for i in range(self.nr_particles):
                    plt.plot(self.historyX[i], self.historyY[i], '.', markevery=[len(self.particle_historyX)-1], label='v = ' + str(self.v[i]) + " \u03BCm/s")
        else:
            plt.plot(self.X[np.where(self.active_particles==1)], self.Y[np.where(self.active_particles==1)], 'o')
            plt.plot(self.X[np.where(self.active_particles==0)], self.Y[np.where(self.active_particles==0)], 'o')

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
        if self.active_particles[i] == 1:
            self.X[i] += self.velocities[i, 0]
            self.Y[i] += self.velocities[i, 1]
            if self.X[i] > self.grid_length:
                self.X[i] = 0
            if self.Y[i] > self.grid_length:
                self.Y[i] = 0
            if self.X[i] < 0:
                self.X[i] = self.grid_length
            if self.Y[i] < 0:
                self.Y[i] = self.grid_length
        (collided, particles_on_site) = self.collision(i)
        if collided and self.time_step > 2:
            self.X[particles_on_site] += self.velocities[i, 0]
            self.Y[particles_on_site] += self.velocities[i, 1]
        self.historyX[i].append(float(self.X[i]))
        self.historyY[i].append(float(self.Y[i]))

    def inactive_noise(self, noise_factor):
        r = np.random.uniform(-noise_factor, noise_factor)

        self.X[np.where(self.active_particles==0)] += r
        self.Y[np.where(self.active_particles==0)] += r

    def compute_MSD(self, i, step):
        MSD = 0
        for t in range(step):
            if t + self.tau < len(self.historyX[0]):
                MSD += ((self.historyX[i][t + self.tau] - self.historyX[i][t])**2 + 
                                (self.historyY[i][t + self.tau] - self.historyY[i][t])**2)
        MSD /= (self.nr_steps - 1 * (self.tau - 1)) 
        self.historyMSD[i].append(MSD)
    
    def compute_torque(self, n):
        active_X = np.delete(self.X, np.where(self.X==self.X[n]))   # n =/= i
        active_Y = np.delete(self.Y, np.where(self.Y==self.Y[n]))   # n =/= i
        active_X = active_X[np.where(self.active_particles==1)[0]]
        active_Y = active_Y[np.where(self.active_particles==1)[0]]

        passive_X = active_X[np.where(self.active_particles==1)[0]]
        passive_Y = active_Y[np.where(self.active_particles==1)[0]]
        
        v_nx = self.velocities[n, 0]
        v_ny = self.velocities[n, 1]
        
        v_n_r_ni = v_nx * np.add(np.abs(np.subtract(self.X[n], active_X)), v_ny * np.abs(np.subtract(self.Y[n], active_Y)))   
        r_ni_2 = np.power(np.add(np.abs(np.subtract(self.X[n], active_X)), np.abs(np.subtract(self.Y[n], active_Y))), 2)
        cross = np.subtract(v_nx * (np.abs(np.subtract(self.Y[n], active_Y))), v_ny * np.abs(np.subtract(self.X[n], active_X)))
        sum1 = self.T0 * np.sum(np.multiply(np.divide(v_n_r_ni, r_ni_2), cross))

        v_n_r_nm = v_nx * np.add(np.abs(np.subtract(self.X[n], passive_X)), v_ny * np.abs(np.subtract(self.Y[n], passive_Y)))   
        r_nm_2 = np.power(np.add(np.abs(np.subtract(self.X[n], passive_X)), np.abs(np.subtract(self.Y[n], passive_Y))), 2)
        cross = np.subtract(v_nx * (np.abs(np.subtract(self.Y[n], passive_Y))), v_ny * np.abs(np.subtract(self.X[n], passive_X)))
        sum2 = self.T0 * np.sum(np.multiply(np.divide(v_n_r_nm, r_nm_2), cross))

        T_n = sum1 - sum2

        if T_n != 0:
            self.theta[n] += T_n
                 
    def step(self, TAU=1, SAVEFIG=False, SAVETHRESH=50, LEGEND=True, MSD=False, SAVE_DATA=False):
        self.initialize_inactive_particles()
        for step in range(self.nr_steps):
            if SAVE_DATA:
                self.X_history.append(self.X)
                self.Y_history.append(self.Y)
            self.inactive_noise(noise_factor=0.01)
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
        distance = np.sqrt(np.add(np.power(self.X, 2), np.power(self.Y, 2)))
        pop_on_site = np.where(distance < self.r_c)[0]
        if len(pop_on_site) > 1:
            return (True, pop_on_site)
        return (False, pop_on_site)
    
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
    brown = BrownianParticles(nr_particles=1000, grid_length=100, nr_steps=1000, D_T=0.085, D_R=0.001, tau=1, DRAW=True, r_c=5, T0=0.75)
    brown.step(SAVEFIG=False, LEGEND=False, SAVE_DATA=False)

def task2_test():
    brown = BrownianParticles(nr_particles=100, grid_length=100, nr_steps=10, D_T=0.0085, D_R=0.00001, tau=1, DRAW=False, r_c=100, T0=0.75)
    brown.step(SAVEFIG=False, LEGEND=False, SAVE_DATA=True)
    #pd.DataFrame(asarray(brown.X_history)).to_csv("HW3/data2/t100000X.csv", header=None)
    xHist = pd.DataFrame(np.reshape(brown.X_history, (brown.nr_particles*brown.nr_steps, 1)), columns=["colummn"])
    xHist.to_csv('listX.csv', index=False)
    yHist = pd.DataFrame(np.reshape(brown.Y_history, (brown.nr_particles*brown.nr_steps, 1)), columns=["colummn"])
    yHist.to_csv('listY.csv', index=False)
    #pd.DataFrame(asarray(brown.Y_history)).to_csv("HW3/data2/t100000Y.csv", header=None)
    X = genfromtxt("listX.csv", delimiter=',')
    Y = genfromtxt("listY.csv", delimiter=',')
    plt.plot(X, Y, linestyle='none', marker='o')
    plt.show()

#save_data()
#task1()
#plot_iterations(SAVE_DATA=False)
task2()