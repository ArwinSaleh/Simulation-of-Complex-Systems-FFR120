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
        self.susceptible = np.ones((nr_agents, 2))  
        self.susceptible[int(0.9 * nr_agents) : nr_agents] = 0    # Initially, 90% of agents are susceptible.
        self.infected = np.zeros((self.nr_agents, 2))    
        self.infected[int(0.9 * nr_agents) : nr_agents] = 1       # Initially, 10% of agents are infected.
        self.recovered = np.zeros((self.nr_agents, 2))    # Initially, no agents have recovered.

    def draw_agents(self):
        sus_index = np.where(self.susceptible == 1)[0]
        inf_index = np.where(self.infected == 1)[0]
        rec_index = np.where(self.recovered == 1)[0]
        plt.scatter(self.x[sus_index], self.y[sus_index], color='blue')
        plt.scatter(self.x[inf_index], self.y[inf_index], color='red')
        plt.scatter(self.x[rec_index], self.y[rec_index], color='green')
        plt.axis([-10, self.grid_length + 10, -10, self.grid_length + 10])
        plt.draw()
        plt.pause(0.001)
        plt.clf()

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
    
    def infect_susceptibles(self, i):
        
        if self.infected[i, 0] == 1 and self.infected[i, 1] == 1:

            if i < self.nr_agents-1:
            
                if self.susceptible[i+1, 0] == 1 and self.susceptible[i, 1] == 1:
                    self.susceptible[i+1, 0] = 0
                    self.susceptible[i, 1] = 0
                    self.infected[i+1, 0] = 1
                    self.infected[i, 1] = 1
                
                if self.susceptible[i, 0] == 1 and self.susceptible[i+1, 1] == 1:
                    self.susceptible[i, 0] = 0
                    self.susceptible[i+1, 1] = 0
                    self.infected[i, 0] = 1
                    self.infected[i+1, 1] = 1

            if i > 0:

                if self.susceptible[i-1, 0] == 1 and self.susceptible[i, 1] == 1:
                    self.susceptible[i-1, 0] = 0
                    self.susceptible[i, 1] = 0
                    self.infected[i-1, 0] = 1
                    self.infected[i, 1] = 1
                
                if self.susceptible[i, 0] == 1 and self.susceptible[i-1, 1] == 1:
                    self.susceptible[i, 0] = 0
                    self.susceptible[i-1, 1] = 0
                    self.infected[i, 0] = 1
                    self.infected[i-1, 1] = 1

    def recover_agent(self, i):
            if self.infected[i, 0] == 1 and self.infected[i, 1] == 1:
                self.infected[i, 0] = 0
                self.infected[i, 1] = 0
                self.recovered[i, 0] = 1
                self.recovered[i, 1] = 1


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
        agent_should_move = self.neumann_probability()
        if agent_should_move:
            for i in range(self.nr_agents):
                self.move_agent(i)
                agent_should_infect = self.infection_probability()
                agent_should_recover = self.recovery_probability()
                if agent_should_infect:
                    self.infect_susceptibles(i)
                if agent_should_recover:
                    self.recover_agent(i)

def main():
    dis = DiseaseSpreading(time_steps=1000, nr_agents=1000, grid_length=100, diffusion_rate=0.8, infection_rate=0.6, recovery_rate=0.01)
    time_step = 0
    nr_sus = np.zeros((dis.time_steps, 1))
    nr_inf = np.zeros((dis.time_steps, 1))
    nr_rec = np.zeros((dis.time_steps, 1))
    time = np.zeros((dis.time_steps, 1))
    while(time_step < dis.time_steps):
        print("Time step = " + str(time_step))
        dis.draw_agents()
        dis.move_agents()
        nr_sus[time_step] = sum(dis.susceptible[:, 0])
        nr_inf[time_step] = sum(dis.infected[:, 0])
        nr_rec[time_step] = sum(dis.recovered[:, 0])
        time[time_step] = time_step
        time_step += 1
        print("Number of infected: " + str(sum(dis.infected[:, 0])))
    plt.plot(time, nr_sus, color='blue', label='Susceptible Agents')
    plt.plot(time, nr_inf, color='red', label='Infected Agents')
    plt.plot(time, nr_rec, color='green', label='Recovered Agents')
    plt.ylabel('Number of agents')
    plt.xlabel('Time Steps')
    plt.legend()
    plt.show()

main()