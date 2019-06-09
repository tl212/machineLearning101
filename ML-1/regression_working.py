import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Cleaning and reading data
data = pd.read_csv('student-mat.csv', sep=';')
# print(data.head())

# Reshaping data frame (cleaning)
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
print(data.head())

# Setting up column 'G3' values as target (grades). Assigning and naming variable as 'predict'
predict = 'G3'

# Assigning 'X' (data - 'G3') and 'y' target ('G3')
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Splitting data into training and testing. s
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Creating training model
trainModel = linear_model.LinearRegression()

# Fitting the train model and checking its accuracy
trainModel.fit(x_train, y_train)
acc = trainModel.score(x_test, y_test)
print(acc)

# Checking coefficient and intercept
print('Coefficient: \n', trainModel.coef_)
print('Intercept: \n', trainModel.intercept_)

# Assigning variable 'predictions' with the test data
predictions = trainModel.predict(x_test)

# Creating a loop to predict
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])













