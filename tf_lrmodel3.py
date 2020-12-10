from sklearn.datasets import make_regression
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SLRA():
    def __init__(self):
        self.w = tf.Variable(tf.random.uniform([1], dtype=tf.double))
        self.b = tf.Variable(tf.zeros([1], dtype=tf.double))
        self.epochs = 100
        self.learning_rate = 0.01
        print(self.w, self.b)

    def buildModel(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))

        optimizer = tf.keras.optimizers.SGD(self.learning_rate)

        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    def fitModel(self, train_x, train_y):
        self.model.fit(train_x, train_y, batch_size=10, epochs=self.epochs, shuffle=True)

    def predictModel(self, train_x):
        return self.model.predict(train_x)

    def evalModel(self, test_x, test_y):
        return self.model.evaluate(test_x, test_y)


X, y = make_regression(n_samples=100, n_features=1, bias=10.0, noise=10.0, random_state=2)
y = np.expand_dims(y, axis=1)

train_x = X[:80]
test_x = X[80:]

train_y = y[:80]
test_y = y[80:]

model = SLRA()

model.buildModel()
model.fitModel(train_x, train_y)

'''plt.scatter(train_x, train_y)
plt.plot(train_x, model.predictModel(train_x), '-r')
plt.show()'''

print(model.evalModel(test_x, test_y))