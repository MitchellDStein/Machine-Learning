# SVR

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv(r"Regression\Support Vector Regression\Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values # we want to keeo X as a matrix/array
y = dataset.iloc[:, 2].values

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random state number is to test with online course

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))

# Fitting SVR Model to the data
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predict a new result with Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualize SVR results
plt.plot(X, y, 'r.', label = "Dataset")
plt.plot(X, regressor.predict(X), 'g-', label = "Predictions")
plt.title("(SVR Model)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend(loc = "upper left")
plt.show

X_grid = np.arange(min(X), max(X), step = 0.1) # creates a vector from min X to max X at 0.1 intervals
X_grid = X_grid.reshape((len(X_grid), 1)) # reshape the vector to an array
plt.plot(X, y, 'r.', label = "")
plt.plot(X_grid, regressor.predict(X_grid), 'g-', label = "")
plt.title("(SVR Model)")
plt.xlabel("")
plt.ylabel("")
plt.legend(loc = "upper left")
plt.show