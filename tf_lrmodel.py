from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np


class SLRA():
    def __init__(self):
        self.w = np.random.rand(2, 1) * 0.001

    def train(self, train_x, train_y):
        self.epochs = 100
        self.learning_rate = 0.1

        loss_mem = []

        # train_y 차원변환
        train_y = np.expand_dims(train_y, axis=1)

        for i in range(self.epochs):
            hypothesis = np.matmul(train_x, self.w)
            error = hypothesis - train_y
            loss = np.mean(error * error) / 2

            loss_mem.append(loss)

            # 공식적용 및 차원유지 그리고 T를 통한 형상유지
            gradient = np.mean(error * train_x, axis=0, keepdims=True).T

            # GDA적용
            self.w -= self.learning_rate * gradient

        return loss_mem

    def test(self, target_x):
        res = np.matmul(target_x, self.w)

        return res

    def pred(self, test_x, test_y):
        test_y = np.expand_dims(test_y, axis=1)

        cal = np.matmul(test_x, self.w)
        error = cal - test_y
        mse = np.mean(error * error)
        rmse = np.sqrt(mse)

        return rmse


X, y = make_regression(n_samples=100, n_features=1, bias=10.0, noise=10.0, random_state=2)

X_bias = np.insert(X, 0, 1, axis=1)

train_x = X_bias[:80]
test_x = X_bias[80:]

train_y = y[:80]
test_y = y[80:]

model = SLRA()
loss_mem = model.train(train_x, train_y)

x_epoch = list(range(len(loss_mem)))

'''plt.plot(x_epoch, loss_mem)
plt.title('Loss plot')
plt.xlabel('epochs')
plt.ylabel('Loss status')

plt.show()'''

'''plt.scatter(train_x[:,1], train_y)
plt.plot(train_x[:,1], model.test(train_x), '-r')
plt.show()'''

print(model.pred(test_x, test_y))