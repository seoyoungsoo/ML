from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

class mlraWithKeras():
    def __init__(self):
        self.epochs = 100
        self.learning_rate = 0.01

    def buildModel(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))

        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    def fitModel(self, x, y):
        histroy = self.model.fit(x, y, batch_size=10, epochs=self.epochs,
                                 shuffle=True)
        return histroy

    def evalModel(self, x, y):
        return self.model.evaluate(x, y)


X, y = make_regression(n_samples=100, n_features=3, bias=10.0, noise=10.0,
                       random_state=1)

y = np.expand_dims(y, axis=1)

train_x = X[:80]
test_x = X[80:]

train_y = y[:80]
test_y = y[80:]

model = mlraWithKeras()
model.buildModel()
history = model.fitModel(train_x, train_y)

hist = pd.DataFrame(history.history)
print(hist)

x_epoch = list(range(len(hist)))
plt.plot(x_epoch, hist['loss'])
plt.show()

model.evalModel(test_x, test_y)