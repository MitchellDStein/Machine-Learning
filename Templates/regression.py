# Regression template 

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv("")
X = dataset.iloc[:, 1:2].values # we want to keeo X as a matrix/array
y = dataset.iloc[:, 2].values

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random state number is to test with online course

# Fitting Regression Model to the data

# Predict a new result with Regression
y_pred = regressor.predict(np.array([[6.5]]))

# Visualize Regression results
plt.plot(X, y, 'r.', label = "")
plt.plot(X_grid, regressor.predict(X)), 'g-', label = "")
plt.title("__________ (Regression Model)")
plt.xlabel("")
plt.ylabel("")
plt.legend(loc = "upper left")
plt.show


# for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), step = 0.1) # creates a vector from min X to max X at 0.1 intervals
X_grid = X_grid.reshape((len(X_grid), 1)) # reshape the vector to an array
plt.plot(X, y, 'r.', label = "")
plt.plot(X_grid, regressor.predict(X_grid)), 'g-', label = "")
plt.title("__________ (Regression Model)")
plt.xlabel("")
plt.ylabel("")
plt.legend(loc = "upper left")
plt.show
