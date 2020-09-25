import numpy as np
from matplotlib import pyplot as plt

a = np.array([1,2,3,4,5])

value,freq = plt.csd(a,a)

plt.plot(freq,value.real)
plt.show()