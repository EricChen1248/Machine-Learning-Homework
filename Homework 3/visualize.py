import matplotlib.pyplot as plt
import numpy as np

ins = np.load('egins.npy')
plt.hist(ins)
plt.show()

Eins = np.load('Eins.npy')
plt.plot(range(Eins.shape[0]), Eins)
plt.show()

Eouts = np.load('Eouts.npy')
plt.plot(range(Eouts.shape[0]), Eouts)
print(Eouts[-1])
plt.show()