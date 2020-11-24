import numpy as np

a = np.random.normal(0, 1)
b = np.array([1, 2 ,3, 3, 5])
c = np.where(b==3)
c= c[0]
print(c[1])
