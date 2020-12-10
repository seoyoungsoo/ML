from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X, y = make_regression(n_samples=100, n_features=1, bias=10.0, noise=10.0, random_state=2)

plt.scatter(X, y, label="data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

X_bias = np.insert(X, 0, 1, axis=1)

train_x = X_bias[:80]
test_x = X_bias[80:]

train_y = y[:80]
test_y = y[80:]