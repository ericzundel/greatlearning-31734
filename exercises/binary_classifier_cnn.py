# run with `uv run exercises/binary_classfier_cnn.py``

# Classify images of cats and dogs
# Sample data:
#  wget https://www.dropbox.com/s/t4pzwpvrzneb190/training_set.zip
#  wget https://www.dropbox.com/s/i37jfni3d29raoc/test_set.zip

from pathlib import Path

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

SCRIPT_DIR = Path(__file__).parent
TRAINING_DATA_DIR = Path(SCRIPT_DIR, "..", "data", "training_set", "training_set")
TEST_DATA_DIR = Path(SCRIPT_DIR, "..", "data", "test_set", "test_set")

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150

NUM_TRAIN_SAMPLE = 100
NUM_VALIDATION_SAMPLES = 100
EPOCHS = 20
BATCH_SIZE = 20

#for file in Path(TRAINING_DATA_DIR, "cats").iterdir():
#    print(f"file: {file}")
