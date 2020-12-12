import numpy as np

target = [0, 1, 2, 1, 0]

mode = np.eye(3)
print(mode)

one_hot_target = mode[target]
print(one_hot_target)