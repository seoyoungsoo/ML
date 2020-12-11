import numpy as np
import matplotlib.pyplot as plt


def hypothesis(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-10, 10, 0.1, dtype=np.float32)

plt.plot(x, hypothesis(x))
plt.show()
