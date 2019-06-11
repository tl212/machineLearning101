import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

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
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open('studentModel.pickle', 'wb') as f:
            pickle.dump(linear, f)'''

pickle_in = open('studentModel.pickle', 'rb')
linear = pickle.load(pickle_in)


# Checking coefficient and intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Assigning variable 'predictions' with the test data
predictions = linear.predict(x_test)

# Creating a loop to predict
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use('ggplot')
pyplot.scatter(data[p], data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()

