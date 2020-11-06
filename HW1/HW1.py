import random as rnd
import matplotlib.pyplot as plt
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
        self.susceptible = np.ones((nr_agents, 1))  
        self.susceptible[int(0.9 * nr_agents) : nr_agents] = 0    # Initially, 90% of agents are susceptible.
        self.infected = np.zeros((self.nr_agents, 1))    
        self.infected[int(0.9 * nr_agents) : nr_agents] = 1       # Initially, 10% of agents are infected.
        self.recovered = np.zeros((self.nr_agents, 1))    # Initially, no agents have recovered.

    def draw_agents(self, DRAW_PATH=False):
        if not DRAW_PATH:
            plt.clf()
        sus_index = np.where(self.susceptible == 1)[0]
        inf_index = np.where(self.infected == 1)[0]
        rec_index = np.where(self.recovered == 1)[0]
        plt.scatter(self.x[sus_index], self.y[sus_index], color='blue')
        plt.scatter(self.x[inf_index], self.y[inf_index], color='red')
        plt.scatter(self.x[rec_index], self.y[rec_index], color='green')
        plt.axis([-10, self.grid_length + 10, -10, self.grid_length + 10])
        plt.title("D = " + str(self.d) + "      beta = " + str(self.beta) + "     gamma = " + str(self.gamma))
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
        
        if self.infected[i]:

            if i < self.nr_agents-1:

                agent_index_1 = self.get_agent_index(self.x[i] + 1, self.y[i])
                agent_index_2 = self.get_agent_index(self.x[i], self.y[i] + 1)
            
                if self.susceptible[agent_index_1] == 1:
                    self.susceptible[agent_index_1]
                    self.infected[agent_index_1] = 1
                
                if self.susceptible[agent_index_2]  == 1:
                    self.susceptible[agent_index_2]  = 0
                    self.infected[agent_index_2] = 1

            if i > 0:

                agent_index_3 = self.get_agent_index(self.x[i] - 1, self.y[i])
                agent_index_4 = self.get_agent_index(self.x[i], self.y[i] - 1)

                if self.susceptible[agent_index_3] == 1:
                    self.susceptible[agent_index_3] = 0
                    self.infected[agent_index_3] = 1
                
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
            if r == 0 and self.x[i] < 100:
                self.x[i] += 1
                found_a_way = True
            elif r == 1 and self.x[i] > 0:
                self.x[i] -= 1
                found_a_way = True
            elif r == 2 and self.y[i] < 100:
                self.y[i] += 1
                found_a_way = True
            elif r == 3 and self.y[i] > 0:
                self.y[i] -= 1
                found_a_way = True

    def move_agents(self):
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

def task1_3():
    dis = DiseaseSpreading(time_steps=1000, nr_agents=1000, grid_length=100, diffusion_rate=0.8, infection_rate=0.6, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((dis.time_steps, 1))
    nr_inf = np.zeros((dis.time_steps, 1))
    nr_rec = np.zeros((dis.time_steps, 1))
    time = np.zeros((dis.time_steps, 1))
    stop = False
    while(time_step < dis.time_steps and not stop):
        print("Time step = " + str(time_step))
        dis.draw_agents()
        dis.move_agents()
        nr_sus[time_step] = sum(dis.susceptible)
        nr_inf[time_step] = sum(dis.infected)
        nr_rec[time_step] = sum(dis.recovered)
        time[time_step] = time_step
        time_step += 1
        print("Number of infected: " + str(sum(dis.infected)))
    plt.figure()
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.show()

def task1_1():
    dis = DiseaseSpreading(time_steps=1000, nr_agents=1, grid_length=5, diffusion_rate=0.8, infection_rate=0.6, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((dis.time_steps, 1))
    nr_inf = np.zeros((dis.time_steps, 1))
    nr_rec = np.zeros((dis.time_steps, 1))
    time = np.zeros((dis.time_steps, 1))
    stop = False
    while(time_step < dis.time_steps and not stop):
        print("Time step = " + str(time_step))
        dis.draw_agents(DRAW_PATH=True)
        dis.move_agents()
        nr_sus[time_step] = sum(dis.susceptible)
        nr_inf[time_step] = sum(dis.infected)
        nr_rec[time_step] = sum(dis.recovered)
        time[time_step] = time_step
        time_step += 1

def task1_2():
    dis = DiseaseSpreading(time_steps=1000, nr_agents=10, grid_length=5, diffusion_rate=0.8, infection_rate=0.6, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((dis.time_steps, 1))
    nr_inf = np.zeros((dis.time_steps, 1))
    nr_rec = np.zeros((dis.time_steps, 1))
    time = np.zeros((dis.time_steps, 1))
    stop = False
    while(time_step < dis.time_steps and not stop):
        print("Time step = " + str(time_step))
        dis.draw_agents()
        dis.move_agents()
        nr_sus[time_step] = sum(dis.susceptible)
        nr_inf[time_step] = sum(dis.infected)
        nr_rec[time_step] = sum(dis.recovered)
        time[time_step] = time_step
        time_step += 1
        print("Number of infected: " + str(sum(dis.infected)))
    plt.figure()
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.show()

#task1_1()
#task1_2()
task1_3()