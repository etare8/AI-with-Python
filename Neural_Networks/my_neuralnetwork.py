# This code is a simple exercise to learn how to use TensorFlow to model a Multilayer Neural Networks
# This exercise aim at defining a Neural Network able to take data from a CSV file
# and recognize counterfeit banknotes after model training. Data are given in the form of 4 input values
# and an output value; well designed NN is able to infer the correct output of input testing data and a comparison
# between the model and the output values of the testing database is returned in form of % accuracy!!!

import pandas as pd
import csv                  # we are able to handle csv files
import tensorflow as tf     # TensorFlow library for Neural Networks
import numpy as np

## NB!!!:  if you use Native Windows, with Python 3.9-3.12 TensorFlow must be the 2.0 Version and you have to download TensorFlow2.10 for the latest version
##         u have to:  - upgrade pip using pip install pip --upgrade
##                     - pip install "numpy<2.0" is required since some functionality of TF can crash for NumPy 1.0 and NumPy 1.0 conflict
##                     - pip install "tensorflow<2.11" to install a version of TF suitable for Native Window
##                     - NB!! if u have an ERROR for Window Long Path Enabled: u have to open PowerShell as administrator and write enable the Long Path reading with:
##                       >> New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
##                       (then repeat tensorflow<2.11 installation)

"""
- AI Neural Network models mathematical functions that map inputs to outputs
    based on the structue and parameters of the network: the network is shaped
    through training on data.

- Multilayer Neural Networks: artificial network with an input layer, an output layer
    and at least an hidden layer for which we do not provide any value. Hidden layers
    perform some actions and propagate the values forward; through hidden layers we are
    able to model non-linear data!!!
    - Backpropagation: it is the main algorithm used for training neural networks with
        hidden layers-->calculates the errors in the output units, calculate the gradient
        descent for the weights of the previous layer, and repeating the process until the
        input layer is reached.

- TensorFlow: it is a Python library for Neural Networks and it contains methods/functions
    for model definition, adding layers, already implemented backpropagation algorithm
"""

# Data import using CSV

with open(r".\5. Neural Networks\banknotes.csv") as input_file:
    database = csv.reader(input_file)
    next(database)
    # we open the file and we store the content in 'database'
    # database is now an object in the form of rows of 4 elements each

    organized_inputs = []  # we want to separate each line of the file into
    organized_outputs = [] # [x1, x2, x3, x4] input and [y1] output

    # In the following we separate input values x from outputs y
    # x = [[x11,x12,x13,x14], [x21,x22,x23,x24],...,[xn1,xn2,xn3,xn4]]
    # y = [y1, y2, y3, y4,...,yn]
    for line in database:
        input_points = [float(line[i]) for i in range(len(line)-1)]
        output_points = [int(line[len(line)-1])]
        organized_inputs.append(input_points)
        organized_outputs.append(output_points)
    
# Let's generate Training and testing datasets using the train_test_split
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(organized_inputs, organized_outputs, train_size=0.5, shuffle=True)
# it automatically separates the database into Training and Testing dataset, with a given training or testing ensamble size, and shuffling if we want

# Define the Model of the Neural Network using keras in TensorFlow
model = tf.keras.models.Sequential() # create a sequential model instance to which we can add optional number of layers

# First Hidden Layer: 8 units
layer1 = tf.keras.layers.Dense(8, input_shape=(4,), activation="relu")

# Second Hidden Layer: 4 units
layer2 = tf.keras.layers.Dense(4, input_shape=(8,), activation="relu")

# Output Layer: 1 unit
layer_out = tf.keras.layers.Dense(1, activation="sigmoid")

# Adding Layers
model.add(layer1)
model.add(layer2)
model.add(layer_out)

# Training Session
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

# Model fitting with N epochs
model.fit(X_train, Y_train, epochs=20)

# Model Evaluation: we can evaluate the accuracy of the Network metrics="accuracy"!!!
model.evaluate(x_test, y_test, verbose=2) # verbosity =0 no printing final result, =1 printing result and lines at the end, =2 printing final result at the end

# We can also give a look at Weights and Biases
# the method get_weight() can be applied to a keras model and returns (in exact order as layers)
# a series of arrays representing the weights of the first layer, the biases of the first layer, and so on
# In the following we print the (4, 8) Matrix of weights from 4 inputs to 8 units layer and the vector of the 8 bias values
weights_in_1 = model.get_weights()[0] # (4, 8) Matrix of Weights
bias_in_1 = model.get_weights()[1]    #(8, ) Vector of bias values
print(f"\nWeights Matrix --> shape {weights_in_1.shape} \n")
print(weights_in_1)
print(f"\nBias Vector --> shape {bias_in_1.shape}\n")
print(bias_in_1)
print("\n")