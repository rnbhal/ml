from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


import pickle
import numpy as np

def to_categorical(y):
    """ custome to_categorical for using birds classifier by ROHIT BHAL"""
    
    y = np.asarray(y, dtype='int32')
    nb_classes = 1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        '1,0 denots that the image is bird & 0,1 denots not a bird'
        'denoting bird as 1 and not a bird as 0'
        if y[i][0] == 1: 
            Y[i] = 1
    return Y


# Data loading and preprocessing

#due to pickle object being encoded in python 2.X version using encoding latin1
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"),encoding='latin1')

"""
each output contains two value and the if [0] is 1 then it is bird otherwise it is not.

Gonna change it to single value using custom to_categorical
"""

# shuffle the data
X, Y = shuffle(X, Y)

Y = to_categorical(Y)
Y_test = to_categorical(Y_test)

# Real-time data preprocessing
# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
# Input is a 32x32 image with 3 color channels (red, green and blue)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)

# 2 output where 1 is bird and 0 isn't bird
network = fully_connected(network, 1, activation='softmax')

network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier_60.0.tfl.ckpt')

model.fit(X, Y, n_epoch=60, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True,
          run_id='bird-classifier')

# Save model when training is complete to a file
model.save("bird-classifier_60.tfl")
print("Network trained and saved as bird-classifier_60.tfl!")

