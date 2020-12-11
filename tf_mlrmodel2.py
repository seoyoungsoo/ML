from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class mlraWithTF():
    def __init__(self):
        self.epochs = 100
        self.learning_rate = 0.01
        # Variable setting
        self.w = tf.Variable(tf.random.uniform([3, 1], dtype=tf.double))  # 3행 1열 matrix 생성
        self.b = tf.Variable(tf.zeros([1], dtype=tf.double))

    def buildModel(self, x, y):
        with tf.GradientTape() as tape:
            hypothesis = tf.matmul(x, self.w) + self.b  # matmul을 사용해 곱 연산 수행
            loss = tf.reduce_mean(tf.square(hypothesis - y))
            loss_w, loss_b = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(loss_w * self.learning_rate)
        self.b.assign_sub(loss_b * self.learning_rate)
        return loss

    def trainModel(self, x, y):
        data = tf.data.Dataset.from_tensor_slices((x, y))
        data = data.shuffle(buffer_size=50).batch(10)

        loss_mem = []
        for e in range(self.epochs):
            for each, (x, y) in enumerate(data):
                loss = self.buildModel(x, y)
            # print('epoch {0}: loss is {1:.4f}'.format(e, float(loss)))
            loss_mem.append(loss)
        return loss_mem

    def evalModel(self, x, y):
        y_hat = tf.matmul(x, self.w) + self.b
        mse = tf.reduce_mean(tf.square(y_hat - y))
        rmse = tf.sqrt(mse)
        return rmse


X, y = make_regression(n_samples=100, n_features=3, bias=10.0, noise=10.0,
                       random_state=1)

# X = np.insert(X, 0, 1, axis=1)
y = np.expand_dims(y, axis=1)

train_x = X[:80]
test_x = X[80:]

train_y = y[:80]
test_y = y[80:]

model = mlraWithTF()
loss_mem = model.trainModel(train_x, train_y)

x_epoch = list(range(len(loss_mem)))

plt.plot(x_epoch, loss_mem)
plt.xlabel('epochs')
plt.ylabel('Loss status')
plt.show()

print(model.evalModel(train_x, train_y).numpy())