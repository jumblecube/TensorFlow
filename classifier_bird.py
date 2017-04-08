from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

# Load the data set
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

# Shuffle the data
X, Y = shuffle(X, Y)

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

# Input is a 32x32 image with 3 color channels (red, green and blue)
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

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='bird-classifier')

# Save model when training is complete to a file
model.save("bird-classifier.tfl")

print("Network trained and saved as bird-classifier.tfl!")

# Retrieve a layer weights, by layer name:
dense1_vars = tflearn.variables.get_layer_variables_by_name('dense_1')
# Get a variable's value, using model `get_weights` method:
print("Dense_1 layer weights:")
print(model.get_weights(dense1_vars[0]))
# Or using generic tflearn function:
print("Dense1 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense1_vars[1]))

# It is also possible to retrieve a layer weights through its attributes `W`
# and `b` (if available).
# Get variable's value, using model `get_weights` method:
print("Dense2 layer weights:")
print(model.get_weights(dense_2.W))
# Or using generic tflearn function:
print("Dense2 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense_2.b))
