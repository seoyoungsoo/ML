import numpy as np


class LogicGate:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def andGate(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        y = np.dot(x, w.T) - self.b
        if y > 0:
            return 1
        else:
            return 0

    def orGate(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        y = np.dot(x, w.T) - self.b
        if y > 0:
            return 1
        else:
            return 0

    def nandGate(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([self.w1, self.w2])
        y = np.dot(x, w.T) - self.b
        if y > 0:
            return 0
        else:
            return 1


'''logic = LogicGate(1, 1, 1.7)

print(logic.andGate(0, 0))
print(logic.andGate(0, 1))
print(logic.andGate(1, 0))
print(logic.andGate(1, 1))'''

'''logic = LogicGate(1, 1, 0.3)

print(logic.orGate(0, 0))
print(logic.orGate(0, 1))
print(logic.orGate(1, 0))
print(logic.orGate(1, 1))'''

'''logic = LogicGate(1, 1, 1.7)

print(logic.nandGate(0, 0))
print(logic.nandGate(0, 1))
print(logic.nandGate(1, 0))
print(logic.nandGate(1, 1))'''

logic1 = LogicGate(1, 1, 1.7)
logic2 = LogicGate(1, 1, 0.3)
logic3 = LogicGate(1, 1, 1.7)


def xor(x1, x2):
    t1 = logic3.nandGate(x1, x2)
    t2 = logic2.orGate(x1, x2)
    result = logic1.andGate(t1, t2)

    return result


print(xor(0, 0))
print(xor(0, 1))
print(xor(1, 0))
print(xor(1, 1))
