# run with `uv run exercises/display_mnist_data.py`
import cv2
from cv2.typing import MatLike
from matplotlib.pylab import ndarray
import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def cvt_to_grayscale(test_case) -> ndarray:
    return test_case * 255


for i in range(1000, 1010):
    cv2.imshow(f"x_test[{i}]", x_test[i])
    # cv2.imshow(f"x_test[{i}]", cvt_to_grayscale(x_test[i]))
cv2.waitKey()
