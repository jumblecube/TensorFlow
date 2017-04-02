from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Decide if an image is a bird')
parser.add_argument('image', type=str, help='The image file to check')
args = parser.parse_args()

model.load("bmodelbird.tfl")

# Load the image file
img = scipy.ndimage.imread(args.image, mode="RGB")

# Scale it to 32x32
img = scipy.misc.imresize(img, (32, 32),
                          interp="bicubic").astype(np.float32,
                                                   casting='unsafe')

# Predict
prediction = model.predict([img])

# Check the result.
is_bird = np.argmax(prediction[0]) == 1

if is_bird:
    print("That's a bird!")
else:
    print("That's not a bird!")
