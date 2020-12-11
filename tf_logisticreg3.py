import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class LogisticWithTF():
    def __init__(self):
        self.epochs = 1000
        self.learning_rate = 0.17
        self.w = tf.Variable(tf.random.normal(shape=[2, 1], dtype=tf.float32))
        self.b = tf.Variable(tf.random.normal(shape=[1], dtype=tf.float32))

    def train_on_batch(self, x, y):  # batch 단위로 학습 할 메인 모델을 정의
        with tf.GradientTape() as tape:
            logit = tf.matmul(x, self.w) + self.b
            hypothesis = tf.sigmoid(logit)
            eps = 1e-10
            loss = -tf.reduce_mean(y * tf.math.log(hypothesis + eps) + (1 - y) *
                                   tf.math.log(1 - hypothesis + eps))
        loss_dw, loss_db = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(self.learning_rate * loss_dw)
        self.b.assign_sub(self.learning_rate * loss_db)

        return loss

    def fitModel(self, x, y):  # train_x, train_y를 가지고 model을 학습하는 메서드
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=16).batch(8)

        loss_mem = []

        for e in range(self.epochs):
            for step, (x, y) in enumerate(dataset):
                loss = self.train_on_batch(x, y)
            loss_mem.append(loss)
        return loss_mem

    def predictModel(self, x):  # train_x를 가지고 학습된 model의 결과에 적용해서 label을 도출하는 메서드
        logit = tf.matmul(x, self.w) + self.b
        hypothesis = tf.sigmoid(logit)
        return hypothesis

    def evalModel(self, x, y):  # test_x, test_y를 가지고 학습된 model에 적용해서 res_y를 구하고 이를 test_y와 비교한 정확도 도출
        logit = tf.matmul(x, self.w) + self.b
        hypothesis = tf.sigmoid(logit)

        res_y = np.round(hypothesis, 0)
        accuracy = np.sum(res_y == y) / len(y)
        return accuracy


train_x = [[1, 1], [1, 3], [2, 2], [2, 4], [3, 1], [3, 3], [4, 2], [4, 4], [5, 7],
           [5, 9], [6, 6], [6, 8], [7, 7], [7, 9], [8, 6], [8, 8]]
train_y = [[0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]]

# 주의점
# numpy로 적용할 때는 계산의 편의성을 위해, weight에 bias를 추가해서 계산, 따라서 train_x에 1로 구성된 열을 하나 추가
train_x = np.array(train_x)
# train_x = np.insert(train_x, 0, 1, axis=1)
train_y = np.array(train_y)

model = LogisticWithTF()
loss_mem = model.fitModel(train_x, train_y)

epochs_x = list(range(len(loss_mem)))
plt.plot(epochs_x, loss_mem)
plt.show()
