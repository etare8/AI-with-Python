# This code is an exercise to practice with some methods provided by TensorFlow concerning
# Convolutional Neural Networks. A Convolutional Neural Network is a NN that exploits
# "convolution" and "pooling" for analyzing images!!!

# I have coded a CNN and trained it using the MNIST database of handwritten digits. The CNN is able to identify the number written by hands
# In the file "my_ConvNN_test.py", I have photographed using iphone a handwritten number 8 and uploaded to see if the herein developed model is able to identify it!!!
# Here the model is written and exported as "my_model.h5", using load_model in keras is possible to import the model in other python files!

"""
- Computer Vision:
    computer vision is the name that is given to a series of different computational methods
    for analysing digital images. Images can be tought as an esamble of pixels characterized
    by a value ranging from 0 to 255 (the RGB system). We can give the complete list of pixels
    of an image as input dataset to a neural network with the aim of training it to be able to
    recognize what is in the image!!! NB: 1) computational cost is high, 2)number of input
    values is very big

- Convolutional Neural Network:
    It is a Neural Network that uses convolution and pooling for analyzing digital images
    
    - "Convolution":
        We can use a filter kernel matrix to transform an initial ensamble of pixels
        into a new image form reduced in the complexity, in this way we are able to transform
        the initial information in a more suitable form for being processed by a NN!
    - "Pooling":
        Subsections and sectors of the initial image pixel matrix can be substituted by
        selecting one single pixel (e.g., the highest value ) for each sector
    - "Flattening":
        the flattening process allows to take the amount of data coming oout from the
        convolution and pooling processes and align them in a single flat input dataset
        to be processed by a NN model!!! 
"""

import tensorflow as tf
import numpy as np

# Import of MNIST database for handritten digits
database = tf.keras.datasets.mnist

# Data preparing
# MNIST is a database of 60'000 images (28x28 pixels in greyscale) and 10'000 test images
# we have to prepare training and testing datasets
(x_train, y_train), (x_test, y_test) = database.load_data() # returns a tuple[any, any] - tuple[any, any]
# x_train.shape[0] = 60'000 / x_train.shape[1] = 28 / x_train.shape[2] = 28 ---> we have 60'000 matrixes with a 28x28 structure

# Normalization of data: we have pixel values from 0 to 1!!!
x_train = x_train / 255.0
x_test = x_test / 255.0
# Binary cross matrixes
y_train = tf.keras.utils.to_categorical(y_train) # Converts a class vector of integers into a binary class matrix
y_test = tf.keras.utils.to_categorical(y_test)

# we have to reshape the input data to match the neural network requirements, we need 60000 matrixes with x=28, y=28, color channels=1 (2DConvolution)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], x_test.shape[2], 1)

#---------------------------------------

# Model of the Convolutional Neural Network
model_CNN = tf.keras.models.Sequential()

# Image Convolution + Pooling (Max-pooling) layers
# 2D convolutional layer that creates a convolution kernel: from 60000 objects creates 32 objects in 26x26 form (final shape (60000, 26, 26, 32)) 
conv_layer = tf.keras.layers.Conv2D(filters=32, # 32 filters, dimensionality of the output space
                                    kernel_size=(3, 3), # 3x3 kernel matrix
                                    activation="relu",
                                    input_shape=(28, 28, 1)) # input reshaping of the x_train

# Max pooling operation for 2D spatial data: we obtain a shape (60000, 13, 13, 32) by pooling the conv_layer
pool_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

# Flattening of data to be suitable as CNN input data: we obtain 60000 arrays of 5408 elements flattened
flattening = tf.keras.layers.Flatten()

# Hidden layers and output layer
layer1 = tf.keras.layers.Dense(units=128, activation="relu")
drop = tf.keras.layers.Dropout(0.5) # float between 0 to 1 of elements to drop
output_layer = tf.keras.layers.Dense(units=10, activation="softmax") # output units for all the 10 digits

# Adding layers
model_CNN.add(conv_layer)
model_CNN.add(pool_layer)
model_CNN.add(flattening)
model_CNN.add(layer1)
model_CNN.add(drop)
model_CNN.add(output_layer)

# Compile and Training of the Convolutional Neural Network
model_CNN.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
model_CNN.fit(x=x_train, y=y_train, epochs=10)

# Model Evaluation Accuracy!!!
model_CNN.evaluate(x=x_test, y=y_test, verbose=2)

# Model Saving for future use (model.predict(x_input))
model_CNN.save("my_model.h5")
