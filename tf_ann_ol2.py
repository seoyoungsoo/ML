import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def ReLU(x):
    return np.maximum(0, x)


class LayerSigmoid:
    def __init__(self, rows, cols):
        np.random.seed(22)
        self.w = np.random.uniform(low=0.0, high=1.0, size=[rows, cols])

    def forward(self, inputs):
        matmul = np.matmul(inputs, self.w)
        return sigmoid(matmul)


class LayerReLU:
    def __init__(self, rows, cols):
        np.random.seed(22)
        self.w = np.random.uniform(low=0.0, high=1.0, size=[rows, cols])

    def forward(self, inputs):
        matmul = np.matmul(inputs, self.w)
        return ReLU(matmul)


inputs = np.array([[1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 1]])

hidden1 = LayerReLU(4, 5)
hidden2 = LayerReLU(5, 3)
output = LayerSigmoid(3, 1)

step1 = hidden1.forward(inputs)
step2 = hidden2.forward(step1)
outputs = output.forward(step2)

print(outputs)
