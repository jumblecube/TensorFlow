from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import h5py
import numpy

# Load the data set
hf1 = h5py.File('X.h5', 'r')
X = numpy.array(hf1['imgs'])
print("X shape ---> ", X.shape)
Y1 = numpy.array(hf1['labels'])
Y = numpy.zeros(shape=(len(Y1), 43))

for i in range(0, len(Y1)-1):
    Y[i][int(Y1[i])-1] = 1

hf2 = h5py.File('X_test.h5', 'r')
X_test = numpy.array(hf2['imgs'])
print("X_test shape ---> ", X_test.shape)
Y1_test = numpy.array(hf2['labels'])
Y_test = numpy.zeros(shape=(len(Y1_test), 43))

for i in range(0, len(Y1_test)-1):
    Y_test[i][int(Y1_test[i])-1] = 1

# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

# Input is a image with 3 color channels (red, green and blue)
input_layer = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug, name='input')

# Step 1: Convolution
conv2d_l1 = conv_2d(input_layer, 120, 5, activation='relu', name='conv2d_l1')

# Step 2: Max pooling
max_pool_l1 = max_pool_2d(conv2d_l1, 2, name='max_pool_l1')

# Step 3: Convolution again
conv2d_l2 = conv_2d(max_pool_l1, 150, 3, activation='relu', name='conv2d_l2')

# Step 4: Max pooling
max_pool_l2 = max_pool_2d(conv2d_l2, 2, name='max_pool_l2')

# Step 5: Convolution yet again
conv2d_l3 = conv_2d(max_pool_l2, 250, 3, activation='relu', name='conv2d_l3')

# Step 6: Max pooling again
max_pool_l3 = max_pool_2d(conv2d_l3, 2, name='max_pool_l3')

# Step 7: Dropout - throw away some data randomly to prevent over-fitting
dropout_1 = dropout(max_pool_l3, 0.5)

# Step 8: Fully-connected 300 node neural network
dense_1 = fully_connected(dropout_1, 300, activation='tanh', name='dense_1')

# Step 9: Dropout - throw away some data randomly to prevent over-fitting
dropout_2 = dropout(dense_1, 0.5)

# Step 8: Fully-connected neural network with two outputs
dense_2 = fully_connected(dropout_2, 43, activation='softmax', name='dense_2')

# Tell tflearn how we want to train the network
network = regression(dense_2, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, name='target')

print('Network ready!!!')

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0)

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=30, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True,
          snapshot_epoch=True,
          run_id='sign-classifier')

# Save model when training is complete to a file
model.save("sign-classifier.tfl")

print("Network trained and saved as sign-classifier.tfl!")
