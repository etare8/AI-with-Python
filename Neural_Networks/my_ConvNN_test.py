import tensorflow as tf
import cv2
import numpy as np

# We load the model developed in the code "my_ConvNN.py", that model is trained an tested to identify handwritten digits using the MNIST database
# In this file I tried to load an image shot with the iphone and actually test if the CNN is able to identify the writted digit!!!

# Model from "my_ConvNN.py"
ConvNN_Model = tf.keras.models.load_model(r".\5. Neural Networks\my_model.h5") # "number_written2.jpg" is equal to number_written.jpg after applying a filter with kernel for borders

# Image preparation:
new_input_image = cv2.imread(r".\5. Neural Networks\number_written2.jpg") # import image
grey_version = cv2.cvtColor(new_input_image, cv2.COLOR_RGB2GRAY)          # converted to greyscale
inverted_image = cv2.bitwise_not(grey_version)                            # after converting to greyscale we have to invert black and white since the MNIST database has black background and white number
resized_image = cv2.resize(inverted_image, (28, 28))                      # image resized in 28x28 format
normalized_image = resized_image / 255.0                                  # normalization of pixels from 0 to 1 values

new_input = normalized_image.reshape(1, 28, 28, 1)

# Making predictions using the Model and the New input
prediction = ConvNN_Model.predict(new_input)

# Printing Results
print("\nPrediction results:\n")
for i in range(11):
    print(f"{i}: ", round(prediction[0][i] * 100, 2), "%")

print("\nIdentified Number: ", np.where(prediction[0] == max(prediction[0]))[0][0], "\n") # since it is an ndarray I have to use np.where logic and not index()

# Image "number_written2.jpg" but with Black and White inverted as explained in line 14 (bitwise_not())
resize_to_show_image = cv2.resize(inverted_image, (500, 500))
cv2.imshow('Image in Negative version!',resize_to_show_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
