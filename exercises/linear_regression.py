# Run with uv run exercises/linear_regression.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Exercise 1 from video "Linear Regression Using Tensorflow"

# Linear model y = w*x + b
# Define model parameters
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Define model inputs
# This will come from the data file
x = tf.Variable([5.0], tf.float32)

# Define model ouptut
y = w * x + b

print(f"w:{w} x:{x} b:{b} y:{y}")


# Build a linear model
train_X = [
    3.3,
    4.4,
    5.5,
    6.71,
    6.93,
    4.168,
    9.779,
    6.182,
    7.69,
    2.167,
    7.042,
    10.791,
    5.313,
    7.997,
    5.654,
    9.27,
    3.1,
]
train_Y = [
    1.7,
    2.76,
    2.09,
    3.19,
    1.694,
    1.573,
    3.366,
    2.596,
    2.53,
    1.221,
    2.827,
    3.456,
    1.65,
    2.904,
    2.42,
    2.94,
    1.3,
]

NUM_EXAMPLES = len(train_X)
NUM_TRAIN_RESULTS = len(train_Y)

print(f"NUM_EXAMPLES: {NUM_EXAMPLES}")
print(f"NUM_TRAIN_RESULTS: {NUM_TRAIN_RESULTS}")
assert NUM_EXAMPLES == NUM_TRAIN_RESULTS

# Create model parameters with starting values:
# this is modeling a single neuron
W = tf.Variable(0.0)  # Weight
b = tf.Variable(0.0)  # Bias

train_steps = 100
learning_rate = 0.01

#
for i in range(train_steps):
    # Watch the gradient flow.
    # The steps inside the loop will be recorded in 'tape'
    #
    # Note: This isn't a great example to follow for the future.
    # Using the tape is a way to use tensorflow 1.0 feature of keeping graphs in Tensorflow 2.0
    with tf.GradientTape() as tape:
        # Forward pass
        yhat = train_X * W + b

        # Calculate error loss
        error = yhat - train_Y
        # tf.square(error) - sum of square of all the errors
        #   You square the error so that outliers are "punished" more than
        #   errors that are close to the predicted line
        #   Also, the error function is turned into a parabola instead of a 'v'
        #   which makes things settle more easily.
        #   See also, gaussian distribution of error
        loss = tf.reduce_mean(tf.square(error))

    # Evaluate the gradient with respect to the parameters
    # Apply the gradient descent using this loss function
    dW, db = tape.gradient(loss, [W, b])

    print("dW: ", dW)
    print("db: ", db)

    # Update the weight based on what the gradient descent tells us
    # assign_sub == subtract another  tensor from this one. A bit silly
    # in this example since both W and dW are actually scalars.
    W.assign_sub(dW * learning_rate)

    # Update the bias based on what the gradient descent tells us
    b.assign_sub(db * learning_rate)

    # print the loss
    if i % 10 == 0:
        print("Loss at pass {:03d}: {:.3f}".format(i, loss))

print(f"W: {W.numpy()}, b: {b.numpy()}")


# Graphic display
plt.plot(train_X, train_Y, "ro", label="Original data")
plt.plot(train_X, np.array(W * train_X + b), label="Fitted line")
plt.legend()
plt.show()


# @title
def linear_model(input_value) -> float:
    return (W * input_value + b).numpy()


new_inputs = []
predictions = []
for i in range(0, 10):
    new_inputs.append(float(i))
    predictions.append(linear_model(i))

plt.plot(new_inputs, predictions, "ro", label="predictions")
plt.show()
