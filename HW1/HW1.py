import random as rnd
import matplotlib.pyplot as plt
import numpy as np

class DiseaseSpreading:
    def __init__(self, nr_agents, grid_length, diffusion_rate, infection_rate, recovery_rate):
        self.x = np.random.randint(0, 100, nr_agents)
        self.y = np.random.randint(0, 100, nr_agents)
        self. nr_agents = nr_agents
        self.grid_length = grid_length
        self.d = diffusion_rate
        self.beta = infection_rate
        self.gamma = recovery_rate
        self.susceptible_x = np.ones((self.nr_agents, 1))  # Initially, every agent is susceptible.
        self.susceptible_y = np.ones((self.nr_agents, 1))  # Initially, every agent is susceptible.
        self.infected_x = np.zeros((self.nr_agents, 1))    # Initially, no agents are infected.
        self.infected_y = np.zeros((self.nr_agents, 1))    # Initially, no agents are infected.
        self.recovered_x = np.zeros((self.nr_agents, 1))   # Initially, no agents have recovered.
        self.recovered_y = np.zeros((self.nr_agents, 1))   # Initially, no agents have recovered.

    def draw_agents(self):
        plt.scatter(self.x, self.y)
        plt.axis([0, self.grid_length, 0, self.grid_length])
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
        
        if i < self.nr_agents-1:
        
            if self.susceptible_x[i+1] == 1 and self.susceptible_y[i] == 1:
                self.susceptible_x[i+1] = 0
                self.susceptible_y[i] = 0
                self.infected_x[i+1] = 1
                self.infected_y[i] = 1
            
            if self.susceptible_x[i] == 1 and self.susceptible_y[i+1] == 1:
                self.susceptible_x[i] = 0
                self.susceptible_y[i+1] = 0
                self.infected_x[i] = 1
                self.infected_y[i+1] = 1

        if i > 0:
            if self.susceptible_x[i-1] == 1 and self.susceptible_y[i] == 1:
                self.susceptible_x[i-1] = 0
                self.susceptible_y[i] = 0
                self.infected_x[i-1] = 1
                self.infected_y[i] = 1
            
            if self.susceptible_x[i] == 1 and self.susceptible_y[i-1] == 1:
                self.susceptible_x[i] = 0
                self.susceptible_y[i-1] = 0
                self.infected_x[i] = 1
                self.infected_y[i-1] = 1

    def recover_agent(self, i):
            if self.infected_x[i] == 1 and self.infected_y[i] == 1:
                self.infected_x[i] = 0
                self.infected_y[i] = 0
                self.recovered_x[i] = 1
                self.recovered_y[i] = 1


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
    dis = DiseaseSpreading(nr_agents=1000, grid_length=100, diffusion_rate=0.5, infection_rate=0.5, recovery_rate=0.5)
    time_step = 1
    while(1):
        print("Time step = " + str(time_step))
        dis.draw_agents()
        dis.move_agents()
        time_step += 1

main()