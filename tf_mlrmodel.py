from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class mlraWithNumpy():
    def __init__(self):
        self.epochs = 100
        self.learning_rate = 0.1
        self.w = np.random.rand(4,1) * 0.01

    def buildModel(self, x, y):
        loss_mem = []
        for e in range(self.epochs):
            hypothesis = np.matmul(x, self.w)
            error = hypothesis - y
            loss = np.mean(error * error) / 2
            loss_mem.append(loss)
            gradient = np.mean(x * error, axis=0, keepdims=True).T
            self.w -= self.learning_rate * gradient

        return loss_mem

    def predictModel(self, x):
        return np.matmul(x, self.w)

    def evalModel(self, x, y):
        hypothesis = np.matmul(x, self.w)
        error = hypothesis - y
        mse = np.mean(error * error)
        return np.sqrt(mse)


X, y = make_regression(n_samples=100, n_features=3, bias=10.0, noise=10.0,
                       random_state=1)

X = np.insert(X, 0, 1, axis=1)
y = np.expand_dims(y, axis=1)

train_x = X[:80]
test_x = X[80:]

train_y = y[:80]
test_y = y[80:]

model = mlraWithNumpy()
loss_mem = model.buildModel(train_x, train_y)

x_epoch = list(range(len(loss_mem)))

plt.plot(x_epoch, loss_mem)
plt.show()

print(model.evalModel(test_x, test_y))
