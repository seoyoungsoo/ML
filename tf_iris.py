import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical


class IrisModel():
    def __init__(self):
        self.epochs = 200
        self.learning_rate = 0.04

    def buildModel(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(3, activation='softmax'))
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=['accuracy'])

    def fitModel(self, x, y):
        hist = self.model.fit(x, y, epochs=self.epochs, batch_size=10, shuffle=True)

        return hist

    def evaluateModel(self, x, y):
        self.model.evaluate(x, y)


dataset = load_iris()

'''f, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(dataset.data[:, 0], dataset.data[:, 1], c=dataset.target)
ax[0].set_title('scatter with Sepal')
ax[1].scatter(dataset.data[:, 2], dataset.data[:, 3], c=dataset.target)
ax[1].set_title('scatter with Petal')
plt.show()
'''

dataset_y = to_categorical(dataset.target)
dataset_y = np.array(dataset_y, dtype=np.int32)

dataset_x = dataset.data
dataset_x, dataset_y = shuffle(dataset_x, dataset_y)

train_x = dataset_x[:120]
test_x = dataset_x[120:]

train_y = dataset_y[:120]
test_y = dataset_y[120:]

model = IrisModel()
model.buildModel()
hist = model.fitModel(train_x, train_y)

plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.show()

model.evaluateModel(test_x, test_y)
