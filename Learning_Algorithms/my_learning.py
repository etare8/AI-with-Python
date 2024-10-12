import csv # import to read csv files
import pandas as pd 
import sklearn # import Machine Learning library
import numpy as np
import matplotlib.pyplot as plt

# This file is a testing file to exercise in unsing AI Supervised Learning Algorithms
# Supervised Learning: Nearest-Neighbors Classification / Perceptron Learning / Support Vector Machines / Linear Regression for Previsions
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors Classifier
from sklearn import svm                             # Support Vector Machine Algorithm
from sklearn.linear_model import Perceptron         # Perceptron Learning
from sklearn.linear_model import LogisticRegression # Logistic Regression applicable to Perceptron learning
from sklearn.linear_model import LinearRegression   # Linear Regression algorithm for forecast

"""
1) K-Nearest Neighbors:
    supervised learning task of learning a function mapping an input point to a discrete category (1 or 0). Given an input set of information
    the machine identifies a specific discrete category associated to it. Once the initial training set with already prefixed/known categories (pattern)
    is given to the machine, the new input data point given is classified looking at its k neighbors!!!

2) Support Vector Machines:
    it is a different approach with respect to Neighbors Algorithms and Linear Regressions that uses an additional vector near the decision boundary to
    make the best decision when separating the data in the two categories! It looks for the Maximum Separator Boundary, namley the boundary which stays
    as far as possible from the two groups of data points, thus optimizing the decision!!! SVM can represent boundaries with more than 2 dimensions and
    also non-linear boundaries: it can be necessary and optimal to be used when we are working with non-linear separators!!!

3) Perceptron Algorithm:
    it is opposite to N-N algorithms since it looks at the dataset as a whole and try to identify a "decision boundary". It uses an hypotesis function to
    represent the real decision boundary where the function is dependent on the input data and also on weights that are continously updated and adjusted
    to make the function more accurate!!! Once the function is defined and we give a new data, a class is assigned to the point (0 or 1): a more
    accuarate result can be given by an hypothesis function that has the shape of a logistic function (logistic regression) able to express also
    confidence levels of the estimate and not only a hard threshold (we can express uncertinty and not just 0 or 1 neatly)!!!

4) Linear Regression (line 105):
    supervised learning task of a function that maps an input to a continous value (real number) that stays on a line which is derived from a model
    developed using known input-output database!!! The hypotesis function will generate a line (linear model) that allows us to predict the output
    value of an input value!!! It is different from previous Algorithms since it is used to predict and not to classify!!
"""

# Import data from CSV file using Pandas library. One can use also csv library!!!
database = pd.read_csv(r".\4. Learning\banknotes.csv", sep=",", decimal=".")
lenght = len(database["variance"])

input_data = []
output_data = []
# Input data are stored as a list of lists while output are stored as a list of 0 or 1 values
for i in range(lenght):
    input_point = [float(database["variance"][i]), float(database["skewness"][i]), float(database["curtosis"][i]), float(database["entropy"][i])]
    output_point = [int(database["class"][i])]
    input_data.append(input_point)
    output_data.append(output_point[0])

# Separate the whole database in a Training and Testing datasets:
# 2 ways:

# 1) Manually:
"""
holdout = int(0.60*lenght) # 40% of the database is for training --> the index that represents the 40%
X_training = input_data[:holdout]
Y_training = output_data[:holdout]
x_testing = input_data[holdout:]
y_testing = output_data[holdout:]
"""

# 2) using train_test_split (we can set directly the percentage to use as test set and also if we want to shuffle data!!!):
X_training, x_testing, Y_training, y_testing = train_test_split(input_data, output_data, test_size=0.4, shuffle=True)

# Create the model we wont to use:
model1 = KNeighborsClassifier(n_neighbors=1)
model2 = svm.SVC()
model3 = Perceptron()
#model3 = LogisticRegression()

# Fit the model with the Training dataset created
model1.fit(X_training, Y_training)
model2.fit(X_training, Y_training)
model3.fit(X_training, Y_training)

# Predict the results using the X_testing and compare model prediction with the actual y_testing value
prediction1 = model1.predict(x_testing)
prediction2 = model2.predict(x_testing)
prediction3 = model3.predict(x_testing)

# Comparison between Actual values stored as y_testing and the model prediction!!
matched_result1 = (y_testing == prediction1).sum()
notmatched_results1 = (y_testing != prediction1).sum()
number_predicted = len(prediction1)

matched_result2 = (y_testing == prediction2).sum()
notmatched_results3 = (y_testing != prediction2).sum()

matched_result3 = (y_testing == prediction3).sum()
notmatched_results3 = (y_testing != prediction3).sum()

# Printing accuracy results:
print(f"Accuracy {type(model1).__name__}: ", round((100*matched_result1/number_predicted), 2), "%")
print(f"Accuracy {type(model2).__name__}: ", round((100*matched_result2/number_predicted), 2), "%")
print(f"Accuracy {type(model3).__name__}: ", round((100*matched_result3/number_predicted), 2), "%")


# Plotting graphical representation (without the 4th variable since is not possble to be represented)
fig1, axes = plt.subplots(subplot_kw={'projection':'3d'})

axes.set_title(f"Classification Algorithm: {type(model1).__name__}")

for i in range(int(0.60*lenght)):
    if Y_training[i] == 1:
        color = 'red'
    else:
     color = 'blue'
    axes.scatter(X_training[i][0], X_training[i][1], X_training[i][2], color=color)

plt.close(fig=fig1)


#---------------------------------------------------------------------------------------------------------
##########################################
## Linear Regression: prediction model  ##
##########################################

# Data geneation in a linear form and separation of Training and Testing data
x_gen = np.random.rand(100, 1)
y_gen = 2*x_gen + np.random.rand(100, 1)

x_train, x_test, y_train, y_test = train_test_split(x_gen, y_gen, train_size=0.5, shuffle=True)

# Model definition
linear_regression = LinearRegression()

# Model fitting
linear_regression.fit(x_train, y_train)

# Model prediction
linear_curve = linear_regression.predict(x_test)

plt.figure(2)
plt.title("Linear Regression of randomly generated points", fontsize=14, fontstyle='italic')
plt.xlabel("input X", fontsize=12)
plt.ylabel("input Y", fontsize=12)
plt.scatter(x_test[:, 0], y_test[:, 0], marker='o', color='red', label="Testing dataset")
plt.scatter(x_train[:, 0], y_train[:, 0], marker='o', color='blue', label="Training dataset")
plt.plot(x_test[:, 0], linear_curve[:, 0], color='blue')
plt.grid("grey", linestyle="--", linewidth=0.8)
plt.legend()

for i in range(len(x_test)):
    plt.vlines(x=x_test[i, 0], ymin=(y_test[i, 0] if y_test[i, 0]<linear_curve[i, 0] else linear_curve[i, 0]),
                ymax=(y_test[i, 0] if y_test[i, 0]>linear_curve[i, 0] else linear_curve[i, 0]), colors='black', linestyles='--', linewidth=0.8)


plt.close()
plt.show()