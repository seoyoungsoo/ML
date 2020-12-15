import numpy as np
import matplotlib.pyplot as plt


def stepAF(x):
    y = x > 0
    return y.astype(np.int)


def sigmoidAF(x):
    return 1 / (1 + np.exp(-x))


def ReLUAF(x):
    return np.maximum(0, x)


def leakyReLUAF(x):
    return np.maximum(0.1*x, x)


def tanhAF(x):
    return np.tanh(x)

# Step Function
'''x = np.arange(-5.0, 5.0, 0.1)
y = stepAF(x)
plt.ylim(-0.1, 1.1)
plt.xlim(-5.1, 5.1)
plt.plot(x, y)
plt.show()'''

# Sigmoid Function
'''x = np.arange(-5.0, 5.0, 0.1)
y = sigmoidAF(x)
plt.ylim(-0.1, 1.1)
plt.xlim(-5.1, 5.1)
plt.plot(x, y)
plt.show()'''

# ReLU Function
'''x = np.arange(-5.0, 5.0, 0.1)
y = ReLUAF(x)
plt.ylim(-0.1, 5.1)
plt.xlim(-5.1, 5.1)
plt.plot(x, y)
plt.show()'''

# leakyReLU Function
'''x = np.arange(-5.0, 5.0, 0.1)
y = leakyReLUAF(x)
plt.ylim(-0.6, 5.1)
plt.xlim(-5.1, 5.1)
plt.plot(x, y)
plt.show()'''

x = np.arange(-5.0, 5.0, 0.1)
y = tanhAF(x)
plt.ylim(-1.1, 1.1)
plt.xlim(-5.1, 5.1)
plt.plot(x, y)
plt.show()