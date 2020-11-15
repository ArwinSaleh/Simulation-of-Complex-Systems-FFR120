import numpy as np

a = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
b = np.where(a == 1)


for pos in b:
    print(pos[0])