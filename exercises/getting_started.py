# Execute with uv run exercises/getting_started.py
import tensorflow as tf
# import numpy as np

x = tf.constant(10.0)
y = tf.constant(15.0)
z = x + y
print(f"z is {z}")
print(z)

# Creating tensors through constants
hello = tf.constant("Hello TensorFlow!!")
print("Hello Numpy: ", hello.numpy())


# Creating tensors through operations
node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)

print(f"Sum of node1 and node2 is: {node3.numpy()}")

A = tf.Variable(
    initial_value=([[0, 1, 2, 3], [5, 6, 7, 8]]), shape=(2, 4), dtype="int32", name="A"
)
print(f"A.numpy()  = {A.numpy()}")  # prints out initial_value
print(f"A.shape    = {A.shape}")  # (2,4)
print(f"tf.rank(A) = {tf.rank(A)}")  # 2
print(tf.rank(A))  # tf.Tensor(2, shape=(), dtype=int32)
# Rank is the number of dimensions for the array
# shape is the how long each dimension is


B = tf.Variable([[1], [2], [3], [4]])
print(f"B.shape    = {B.shape}")  # (4,1)
print(f"tf.rank(B) = {tf.rank(B)}")  # tf.rank(B) = 2
print(tf.rank(B))  # tf.Tensor(2, shape=(), dtype=int32)

C = tf.Variable([[1, 2], [3, 4]])
print(f"C.shape    = {C.shape}")  # (4,1)
print(f"tf.rank(C) = {tf.rank(C)}")  # tf.rank(B) = 2
print(tf.rank(C))  #
print(C)  #

# Bias layers are constant types
D = tf.constant(50, shape=[6, 2])
print("rank of D", tf.rank(D))  # tf.Tensor(2, shape(), dtype=int32)
print("shape of D", D.shape)  # (6,2)
print(
    "content of D", D
)  # [[50 50] [50 50] [50 50] [50 50] [50 50] [50 50]], shape=(6, 2), dtype=int32)

# Creating tensors from Exisiting Objects
tf_t = tf.convert_to_tensor(5.0, dtype=tf.float64)
print(
    "tf.rank(tf_t): ", tf.rank(tf_t)
)  # tf.rank(tf_t):  tf.Tensor(0, shape=(), dtype=int32)
print(
    "tf.shape(tf_t):", tf.shape(tf_t)
)  # tf.shape(tf_t): tf.Tensor([], shape=(0,), dtype=int32)
