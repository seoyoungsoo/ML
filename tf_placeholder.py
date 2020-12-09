import tensorflow as tf


@tf.function
def adder(a, b):
    return a + b


A = tf.constant(1)
B = tf.constant(2)
tf.print(adder(A, B))


C = tf.constant([1, 3])
D = tf.constant([2, 4])

tf.print(adder(C, D))
