1. Digit classification using the publicly available MNIST dataset
2. Image classification using Convolutional Neural Networks (CNNs)

The script creates a set of layers that flattens the 28x28 mnist inputs to a 1D array,
then has a layer of interconnected neurons,
then has a dropout layer to prevent overtraining,
then has 10 neurons for outputs for the digits 0-9

Overfitting:  Too much training - the model becomes too specific to your training set.

Training with 5 epochs shows .9775 accuracy
