# run with `uv run exercises/binary_classfier_cnn.py``

# Classify images of cats and dogs
# Sample data:
#  wget https://www.dropbox.com/s/t4pzwpvrzneb190/training_set.zip
#  wget https://www.dropbox.com/s/i37jfni3d29raoc/test_set.zip

from pathlib import Path
from os import dirname

import tensorflow
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
    Activation,
)
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plot
import matplotlib.image as mpimage

SCRIPT_DIR = Path(dirname(__file__))
TRAINING_DATA_DIR = Path(SCRIPT_DIR, "..", "data", "training_set", "training_set")
TEST_DATA_DIR = Path(SCRIPT_DIR)
