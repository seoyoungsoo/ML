from tensorflow.keras.utils import to_categorical

target = [0, 1, 2, 1, 0]
one_hot_target = to_categorical(target)
print(one_hot_target)