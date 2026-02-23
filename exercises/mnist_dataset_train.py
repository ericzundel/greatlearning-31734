import tensorflow as tf
import numpy as py
import matplotlib.pyplot as plot

# Load the model
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert values from 0-255 integers to 0.0 to 1.0
x_train, x_test = x_train /255.0, x_test / 255.0

# Build the model
model = tf.keras.models.Sequential([
    # Layer 1: change 2D array to 1D array
    tf.keras.layers.Flatten(input_shape=(28,28)),
    # Create a layer of fully connected neurons.
    # activation means activation is the element-wise activation function
    # relu means: Rectified Linear Unit activation, aka only activate on positive input
    tf.keras.layers.Dense(128,activation='relu'),

    # This is a regularization tehchnique, used during training to prevent a neural network
    # from overfitting (memorizing the input data).
    # .2 means randomly ignore 20% of the input neurons.
    tf.keras.layers.Dropout(0.2),

    # Output layer, one neuron for each digit.
    tf.keras.layers.Dense(10),
])

# Let's take a look at running the model on the first image in the training set
predictions = model(x_train[:1]).numpy()
# This returns a value for each neuron in the output. Note that some are negative
print ("predictions: ", predictions)

# We can get the probability of each class instead of just the values of the vector
# softmax will convert to a floating point value between 0-1.0
value = tf.nn.softmax(predictions).numpy()
print ("after softmax: ", value)


# How to comput loss for each example
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print("Computed loss: ", loss_fn(y_train[:1], predictions).numpy())

# Model complation
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# I0000 00:00:1771853092.359946 3996955 device_compiler.h:196] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.


# Model fitting to minimize loss (training)
# epochs = number of times to train the model.
model.fit(x_train, y_train, epochs=5)

# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 877us/step - accuracy: 0.9135 - loss: 0.2980
# Epoch 2/5
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 780us/step - accuracy: 0.9564 - loss: 0.1469
# Epoch 3/5
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 915us/step - accuracy: 0.9673 - loss: 0.1098
# Epoch 4/5
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 877us/step - accuracy: 0.9717 - loss: 0.0891
# Epoch 5/5
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 779us/step - accuracy: 0.9762 - loss: 0.0747
# 2026-02-23 08:25:00.616660: I external/local_xla/xla/stream_executor/cuda/subprocess_compilation.cc:346] ptxas warning : Registers are spilled to local memory in function 'gemm_fusion_dot_34', 20 bytes spill stores, 20 bytes spill loads


# Test the model to se4e if it works
model.evaluate(x_test, y_test, verbose=2)
# Output:
# 313/313 - 1s - 3ms/step - accuracy: 0.9783 - loss: 0.0715
#
# 97.8% accuracy

model.save('models/mnist_5_epochs.keras')

# Now, in the video, they keep training it
model.fit(x_train, y_train, epochs=20)

# Output:
# Epoch 1/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9795 - loss: 0.0658
# Epoch 2/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 645us/step - accuracy: 0.9809 - loss: 0.0591
# Epoch 3/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 653us/step - accuracy: 0.9825 - loss: 0.0535
# Epoch 4/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 862us/step - accuracy: 0.9837 - loss: 0.0482
# Epoch 5/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9851 - loss: 0.0451
# Epoch 6/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 792us/step - accuracy: 0.9863 - loss: 0.0416
# Epoch 7/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9866 - loss: 0.0397
# Epoch 8/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9877 - loss: 0.0369
# Epoch 9/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 668us/step - accuracy: 0.9886 - loss: 0.0333
# Epoch 10/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 883us/step - accuracy: 0.9883 - loss: 0.0323
# Epoch 11/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9891 - loss: 0.0322
# Epoch 12/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 844us/step - accuracy: 0.9902 - loss: 0.0287
#
# !!! NB(zundel) In the next iteration, loss actually goes up!
# Epoch 13/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9895 - loss: 0.0301
# Epoch 14/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9909 - loss: 0.0257
# Epoch 15/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9908 - loss: 0.0259
# Epoch 16/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 716us/step - accuracy: 0.9911 - loss: 0.0255
# Epoch 17/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 841us/step - accuracy: 0.9914 - loss: 0.0250
# Epoch 18/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 873us/step - accuracy: 0.9918 - loss: 0.0237
# Epoch 19/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 845us/step - accuracy: 0.9917 - loss: 0.0245
# Epoch 20/20
# 1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 790us/step - accuracy: 0.9922 - loss: 0.0230

# Test the model to see if it seems better
# (This is an example of overfitting)
model.evaluate(x_test, y_test, verbose=2)

# 313/313 - 0s - 614us/step - accuracy: 0.9797 - loss: 0.0913
# Note that accuracy is barely imporved at 98%

model.save('models/mnist_20_epochs.keras')