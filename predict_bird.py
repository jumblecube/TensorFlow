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
import tensorflow as tf

parser = argparse.ArgumentParser(description='Decide if an image is a bird')
parser.add_argument('image', type=str, help='The image file to check')
args = parser.parse_args()

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

input_layer = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug, name='input')

# Step 1: Convolution
conv2d_l1 = conv_2d(input_layer, 32, 3, activation='relu', name='conv2d_l1')

# Step 2: Max pooling
max_pool_l1 = max_pool_2d(conv2d_l1, 2, name='max_pool_l1')

# Step 3: Convolution again
conv2d_l2 = conv_2d(max_pool_l1, 64, 3, activation='relu', name='conv2d_l2')

# Step 4: Convolution yet again
conv2d_l3 = conv_2d(conv2d_l2, 64, 3, activation='relu', name='conv2d_l3')

# Step 5: Max pooling again
max_pool_l2 = max_pool_2d(conv2d_l3, 2, name='max_pool_l2')

# Step 6: Fully-connected 512 node neural network
dense_1 = fully_connected(max_pool_l2, 512, activation='relu', name='dense_1')

# Step 7: Dropout - throw away some data randomly to prevent over-fitting
dropout_1 = dropout(dense_1, 0.5)

# Step 8: Fully-connected neural network with two outputs
dense_2 = fully_connected(dropout_1, 2, activation='softmax', name='dense_2')

# Tell tflearn how we want to train the network
network = regression(dense_2, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0,
                    checkpoint_path='bird-classifier.tfl.ckpt')


model.load("bird-classifier.tfl")

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
    print("That's a bird!", prediction)
else:
    print("That's not a bird!", prediction)
