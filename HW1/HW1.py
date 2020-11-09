import random as rnd
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw
import numpy as np

class DiseaseSpreading:
    def __init__(self, time_steps, nr_agents, grid_length, diffusion_rate, infection_rate, recovery_rate):
        self.x = np.random.randint(0, grid_length, nr_agents)
        self.y = np.random.randint(0, grid_length, nr_agents)
        self.time_steps = time_steps
        self.nr_agents = nr_agents
        self.grid_length = grid_length
        self.d = diffusion_rate
        self.beta = infection_rate
        self.gamma = recovery_rate
        self.susceptible = 0
        self.infected = 0
        self.recovered = 0

    def initialize_agents(self, distribution_index=0.9):
        self.susceptible = np.ones((self.nr_agents, 1))  
        self.susceptible[int(distribution_index * self.nr_agents) : self.nr_agents] = 0    # Initially, 90% (default) of agents are susceptible.
        self.infected = np.zeros((self.nr_agents, 1))    
        self.infected[int(distribution_index * self.nr_agents) : self.nr_agents] = 1       # Initially, 10% (default) of agents are infected.
        self.recovered = np.zeros((self.nr_agents, 1))    # Initially, no agents have recovered.

    def initialize_agents_at_center(self):

        self.susceptible = np.ones((self.nr_agents, 1))
        self.infected = np.zeros((self.nr_agents, 1))
        self.recovered = np.zeros((self.nr_agents, 1))
        
        for x in range(int(2*self.grid_length/5), int(3*self.grid_length/5)):
            for y in range(int(2*self.grid_length/5), int(3*self.grid_length/5)):
                agent_index = self.get_agent_index(x, y)
                if agent_index != -1:
                    self.susceptible[agent_index] = 0
                    self.infected[agent_index] = 1  

    def draw_agents(self, current_time, DRAW_PATH=False):
        sus_index = np.where(self.susceptible == 1)[0]
        inf_index = np.where(self.infected == 1)[0]
        rec_index = np.where(self.recovered == 1)[0]
        if DRAW_PATH:
            plt.plot(self.x[sus_index], self.y[sus_index], linestyle='', color='blue', marker='o', markersize=5, alpha=0.2, label='Susceptible')
            plt.plot(self.x[inf_index], self.y[inf_index], linestyle='', color='red', marker='o', markersize=5, alpha=0.2, label='Infected')
            plt.plot(self.x[rec_index], self.y[rec_index], linestyle='', color='green', marker='o', markersize=5, alpha=0.2, label='Recovered')
        else:
            plt.clf()
            plt.scatter(self.x[sus_index], self.y[sus_index], color='blue', label='Susceptible')
            plt.scatter(self.x[inf_index], self.y[inf_index], color='red', label='Infected')
            plt.scatter(self.x[rec_index], self.y[rec_index], color='green', label='Recovered')
        plt.axis([-2, self.grid_length + 2, -2, self.grid_length + 2])
        plt.title("Diffusion Rate = " + str(self.d) + "      Infection Rate = " + str(self.beta) + "     Recovery Rate = " + str(self.gamma) 
        + "\nTime Step: " + str(current_time + 1))
        if not DRAW_PATH:
            plt.legend(bbox_to_anchor=(1, 1))
        plt.draw()
        plt.pause(0.001)

    def neumann_probability(self):
        r = rnd.uniform(0, 1)
        if r <= self.d:
            return True
        return False
    
    def infection_probability(self):
        r = rnd.uniform(0, 1)
        if r <= self.beta:
            return True
        return False
    
    def recovery_probability(self):
        r = rnd.uniform(0, 1)
        if r <= self.gamma:
            return True
        return False

    def get_agent_index(self, x_key, y_key):
        index = -1
        x_tmp = np.where(self.x == x_key)
        y_tmp = np.where(self.y == y_key)
        agent_index = np.intersect1d(x_tmp, y_tmp)
        if len(agent_index) > 0:
            index = agent_index[0]
        return index
    
    def infect_susceptibles(self, i):
        
        if self.infected[i] == 1:

            if i < self.nr_agents-1:

                agent_index_1 = self.get_agent_index(self.x[i] + 1, self.y[i])
                agent_index_2 = self.get_agent_index(self.x[i], self.y[i] + 1)

                if agent_index_1 != -1:
                    if self.susceptible[agent_index_1] == 1:
                        self.susceptible[agent_index_1] = 0
                        self.infected[agent_index_1] = 1

                if agent_index_2 != -1:
                    if self.susceptible[agent_index_2]  == 1:
                        self.susceptible[agent_index_2]  = 0
                        self.infected[agent_index_2] = 1

            if i > 0:

                agent_index_3 = self.get_agent_index(self.x[i] - 1, self.y[i])
                agent_index_4 = self.get_agent_index(self.x[i], self.y[i] - 1)

                if agent_index_3 != -1:
                    if self.susceptible[agent_index_3] == 1:
                        self.susceptible[agent_index_3] = 0
                        self.infected[agent_index_3] = 1
                
                if agent_index_4 != -1:
                    if self.susceptible[agent_index_4] == 1:
                        self.susceptible[agent_index_4] = 0
                        self.infected[agent_index_4] = 1

    def recover_agent(self, i):
            if self.infected[i]:
                self.infected[i] = 0
                self.recovered[i] = 1

    def move_agent(self, i):
        found_a_way = False
        while(found_a_way == False):
            r = rnd.randint(0, 4)
            if r == 0 and self.x[i] < self.grid_length:
                self.x[i] += 1
                found_a_way = True
            elif r == 1 and self.x[i] > 0:
                self.x[i] -= 1
                found_a_way = True
            elif r == 2 and self.y[i] < self.grid_length:
                self.y[i] += 1
                found_a_way = True
            elif r == 3 and self.y[i] > 0:
                self.y[i] -= 1
                found_a_way = True

    def step(self):
        for i in range(self.nr_agents):
            agent_should_move = self.neumann_probability()
            agent_should_infect = self.infection_probability()
            agent_should_recover = self.recovery_probability()
            if agent_should_move:
                self.move_agent(i)
            if agent_should_infect:
                self.infect_susceptibles(i)
            if agent_should_recover:
                self.recover_agent(i)



def task1_1():
    SIR = DiseaseSpreading(time_steps=1000, nr_agents=1, grid_length=100, diffusion_rate=0.8, infection_rate=0.6, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((SIR.time_steps, 1))
    nr_inf = np.zeros((SIR.time_steps, 1))
    nr_rec = np.zeros((SIR.time_steps, 1))
    time = np.zeros((SIR.time_steps, 1))
    stop = False
    SIR.initialize_agents()
    while(time_step < SIR.time_steps and not stop):
        SIR.draw_agents(DRAW_PATH=True, current_time=time_step)
        SIR.step()
        nr_sus[time_step] = sum(SIR.susceptible)
        nr_inf[time_step] = sum(SIR.infected)
        nr_rec[time_step] = sum(SIR.recovered)
        time[time_step] = time_step
        time_step += 1

def task1_2():
    SIR = DiseaseSpreading(time_steps=500, nr_agents=100, grid_length=10, diffusion_rate=0.6, infection_rate=0.1, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((SIR.time_steps, 1))
    nr_inf = np.zeros((SIR.time_steps, 1))
    nr_rec = np.zeros((SIR.time_steps, 1))
    time = np.zeros((SIR.time_steps, 1))
    stop = False
    SIR.initialize_agents()
    while(time_step < SIR.time_steps and not stop):
        SIR.draw_agents(current_time=time_step)
        SIR.step()
        nr_sus[time_step] = sum(SIR.susceptible)
        nr_inf[time_step] = sum(SIR.infected)
        nr_rec[time_step] = sum(SIR.recovered)
        time[time_step] = time_step
        time_step += 1
        if time_step == 50:
            plt.savefig('task1_2_popAt50')
        if time_step == 100:
            plt.savefig('task1_2_popAt100')
        if time_step == 500:
            plt.savefig('task1_2_popAt500')
    plt.figure()
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.title("Diffusion Rate = " + str(SIR.d) + "      Infection Rate = " + str(SIR.beta) + "     Recovery Rate = " + str(SIR.gamma))
    plt.show()

def task1_3():
    SIR = DiseaseSpreading(time_steps=1000, nr_agents=1000, grid_length=100, diffusion_rate=0.4, infection_rate=0.3, recovery_rate=0.1)
    time_step = 0
    nr_sus = np.zeros((SIR.time_steps, 1))
    nr_inf = np.zeros((SIR.time_steps, 1))
    nr_rec = np.zeros((SIR.time_steps, 1))
    time = np.zeros((SIR.time_steps, 1))
    stop = False
    SIR.initialize_agents_at_center()
    while(time_step < SIR.time_steps and not stop):
        SIR.draw_agents(current_time=time_step)
        SIR.step()
        nr_sus[time_step] = sum(SIR.susceptible)
        nr_inf[time_step] = sum(SIR.infected)
        nr_rec[time_step] = sum(SIR.recovered)
        time[time_step] = time_step
        time_step += 1
        if (time_step == 100):
            plt.savefig('task1_3_popAt100')
    plt.figure()
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.title("Diffusion Rate = " + str(SIR.d) + "      Infection Rate = " + str(SIR.beta) + "     Recovery Rate = " + str(SIR.gamma))
    plt.show()

def task2_1():
    SIR = DiseaseSpreading(time_steps=500, nr_agents=1000, grid_length=100, diffusion_rate=0.8, infection_rate=0.6, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((SIR.time_steps, 1))
    nr_inf = np.zeros((SIR.time_steps, 1))
    nr_rec = np.zeros((SIR.time_steps, 1))
    time = np.zeros((SIR.time_steps, 1))
    SIR.initialize_agents_at_center()
    while(time_step < SIR.time_steps):
        SIR.draw_agents(current_time=time_step)
        SIR.step()
        nr_sus[time_step] = sum(SIR.susceptible)
        nr_inf[time_step] = sum(SIR.infected)
        nr_rec[time_step] = sum(SIR.recovered)
        time[time_step] = time_step
        time_step += 1
        if (time_step == 100):
            plt.savefig('task2_1_popAt100')
        if (time_step == 400):
            plt.savefig('task2_1_popAt400')
    plt.figure()
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.title("Diffusion Rate = " + str(SIR.d) + "      Infection Rate = " + str(SIR.beta) + "     Recovery Rate = " + str(SIR.gamma))
    plt.show()

def task2_2():
    SIR = DiseaseSpreading(time_steps=100, nr_agents=1000, grid_length=100, diffusion_rate=0.6, infection_rate=0.9, recovery_rate=0.1)
    time_step = 0
    nr_sus = np.zeros((SIR.time_steps, 1))
    nr_inf = np.zeros((SIR.time_steps, 1))
    nr_rec = np.zeros((SIR.time_steps, 1))
    time = np.zeros((SIR.time_steps, 1))
    SIR.initialize_agents_at_center()
    while(time_step < SIR.time_steps):
        SIR.draw_agents(current_time=time_step)
        SIR.step()
        nr_sus[time_step] = sum(SIR.susceptible)
        nr_inf[time_step] = sum(SIR.infected)
        nr_rec[time_step] = sum(SIR.recovered)
        time[time_step] = time_step
        time_step += 1
        if (time_step == 50):
            plt.savefig('task2_2_popAt50')
        if (time_step == 100):
            plt.savefig('task2_2_popAt100')
        if (time_step == 400):
            plt.savefig('task2_2_popAt400')
    plt.figure()
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.title("Diffusion Rate = " + str(SIR.d) + "      Infection Rate = " + str(SIR.beta) + "     Recovery Rate = " + str(SIR.gamma))
    plt.show()

def task3():
    SIR = DiseaseSpreading(time_steps=1000, nr_agents=1000, grid_length=100, diffusion_rate=0.6, infection_rate=0.8, recovery_rate=0.1)
    run = 0
    gammas = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.020, 0.030, 0.040, 0.050, 0.100, 0.200]
    R_infinity = np.zeros((len(gammas), 1))
    for gamma in gammas:
        R_infinity_reps = np.zeros((10, 1))
        for rep in range(10):
            stop = False
            time = np.zeros((SIR.time_steps, 1))
            nr_rec = np.zeros((SIR.time_steps, 1))
            SIR.initialize_agents(distribution_index=0.99)
            SIR.gamma = gamma
            time_step = 0
            while(time_step < SIR.time_steps and stop == False):
                print(sum(SIR.infected))
                print("Time Step = " + str(time_step))
                print("beta = " + str(SIR.beta))
                print("gamma = " + str(SIR.gamma))
                SIR.step()
                time[time_step] = time_step
                nr_rec[time_step] = sum(SIR.recovered)
                time_step += 1
                if (time_step > 10):
                    if sum(SIR.infected) == 0:
                        stop = True

            R_infinity_reps[rep] = sum(SIR.recovered)

        R_infinity[run] = sum(R_infinity_reps) / len(R_infinity_reps)
        run += 1
    beta_gamma = np.divide(SIR.beta, gammas)
    plt.figure()
    plt.title("Diffusion Rate = " + str(SIR.d) + "      Beta = " + str(SIR.beta) + "     Gammas = " + str(gammas))
    plt.plot(beta_gamma, R_infinity, color='green', label='Recovered Agents')
    plt.ylabel('R infinity (Average of 10 runs)')
    plt.xlabel('k = Beta / Gamma')
    plt.legend()
    plt.show()

#task1_1()
#task1_2()
#task1_3()

#task2_1()
#task2_2()

task3()