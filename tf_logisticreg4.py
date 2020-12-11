import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class LogisticWithKeras():
    def __init__(self):
        self.epochs = 500
        self.learning_rate = 0.17

    def buildModel(self):  # Layer를 기준으로 모델을 정의
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer,
                           metrics=['binary_accuracy'])

    def fitModel(self, x, y):  # train_x, train_y를 가지고 model을 학습하는 메서드
        self.model.fit(x, y, epochs=self.epochs, batch_size=8, shuffle=True)

    def predictModel(self, x):  # train_x를 가지고 학습된 model의 결과에 적용해서 label을 도출하는 메서드
        return self.model.predict(x)

    def evalModel(self, x, y):  # test_x, test_y를 가지고 학습된 model에 적용해서 res_y를 구하고 이를 test_y와 비교한 정확도 도출
        return self.model.evaluate(x, y)


train_x = [[1, 1], [1, 3], [2, 2], [2, 4], [3, 1], [3, 3], [4, 2], [4, 4], [5, 7],
           [5, 9], [6, 6], [6, 8], [7, 7], [7, 9], [8, 6], [8, 8]]
train_y = [[0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]]

# 주의점
# numpy로 적용할 때는 계산의 편의성을 위해, weight에 bias를 추가해서 계산, 따라서 train_x에 1로 구성된 열을 하나 추가
train_x = np.array(train_x)
# train_x = np.insert(train_x, 0, 1, axis=1)
train_y = np.array(train_y)

model = LogisticWithKeras()
model.buildModel()
model.fitModel(train_x, train_y)

'''res_y = model.predictModel(train_x)
plt.scatter(train_x[:, 0:1], train_x[:, 1:2], c=res_y)
plt.show()'''

test_x = [[2, 3], [3, 2], [6, 9], [7, 8], [8, 7]]
test_y = [[0], [0], [1], [1], [1]]

test_x = np.array(test_x, dtype=np.float32)
test_y = np.array(test_y, dtype=np.float32)

print(model.evalModel(test_x, test_y))
