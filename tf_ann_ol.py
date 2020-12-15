import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    x_exp = np.exp(x)
    x_exp_sum = np.sum(x_exp)
    return x_exp / x_exp_sum


def softmax_eInf(x):
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    x_exp_sum = np.sum(x_exp)
    return x_exp / x_exp_sum


'''x = np.arange(-5.0, 5.0, 0.1)
y = np.exp(x)

plt.plot(x, y)
plt.show()'''

# Infinity 발생 확인
'''temp = np.array([100, 500, 1000])
y_hat = np.exp(temp)
print(y_hat)'''

x = np.array([1, 2, 3])
print(softmax_eInf(x))