import numpy as np
import random as rnd

nr_agents = 10

susceptible = np.ones((nr_agents, 2))  
susceptible[int(0.9 * nr_agents) : nr_agents] = 0    # Initially, 90% of agents are susceptible.

for i in range(len(susceptible)):
    print(susceptible[i,:])