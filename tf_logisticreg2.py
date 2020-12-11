import numpy as np
import matplotlib.pyplot as plt


class LogisticWithNumpy():
    def __init__(self):
        self.epochs = 20000
        self.learning_rate = 0.17

    def fitModel(self, x, y):  # train_x, train_y를 가지고 model을 학습하는 메서드
        self.w = np.random.rand(3, 1) * 0.01
        eps = 1e-10

        loss_mem = []

        for e in range(self.epochs):
            logit = np.matmul(x, self.w)
            hypothesis = 1 / (1 + np.exp(-logit))
            loss = y * np.log(hypothesis + eps) + (1 - y) * np.log(1 - hypothesis + eps)
            loss = -np.sum(loss)
            loss_mem.append(loss)
            gradient = np.mean((hypothesis - y) * x, axis=0, keepdims=True).T
            self.w -= self.learning_rate * gradient

        return loss_mem

    def predictModel(self, x):  # train_x를 가지고 학습된 model의 결과에 적용해서 label을 도출하는 메서드
        logit = np.matmul(x, self.w)
        hypothesis = 1 / (1 + np.exp(-logit))
        return hypothesis

    def evalModel(self, x, y):  # test_x, test_y를 가지고 학습된 model에 적용해서 res_y를 구하고 이를 test_y와 비교한 정확도 도출
        logit = np.matmul(x, self.w)
        hypothesis = 1 / (1 + np.exp(-logit))
        res_y = np.round(hypothesis, 0)
        accuracy = np.sum(res_y == y) / len(y)

        return accuracy


train_x = [[1, 1], [1, 3], [2, 2], [2, 4], [3, 1], [3, 3], [4, 2], [4, 4], [5, 7],
           [5, 9], [6, 6], [6, 8], [7, 7], [7, 9], [8, 6], [8, 8]]
train_y = [[0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]]

# 주의점
# numpy로 적용할 때는 계산의 편의성을 위해, weight에 bias를 추가해서 계산, 따라서 train_x에 1로 구성된 열을 하나 추가
# train_x = np.array(train_x)
train_x = np.insert(train_x, 0, 1, axis=1)
train_y = np.array(train_y)

model = LogisticWithNumpy()
loss_mem = model.fitModel(train_x, train_y)

'''epochs_x = list(range(len(loss_mem)))
plt.plot(epochs_x, loss_mem)
plt.show()'''

'''res_y = model.predictModel(train_x)
plt.scatter(train_x[:, 1:2], train_x[:, 2:3], c=res_y)
plt.show()'''

test_x = [[2, 3], [3, 2], [6, 9], [7, 8], [8, 7]]
test_y = [[0], [0], [1], [1], [1]]

test_x = np.insert(test_x, 0, 1, axis=1)

accuracy = model.evalModel(test_x, test_y)
print(accuracy)
