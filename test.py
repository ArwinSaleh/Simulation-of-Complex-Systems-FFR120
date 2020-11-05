import numpy as np
import random as rnd

susceptible = np.ones((10, 2))  
susceptible[int(0.9 * 10) : 10] = 0    # Initially, 90% of agents are susceptible.

print(susceptible[2, 1])