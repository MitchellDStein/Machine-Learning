# Multiple Linear Regression

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv("Regression/Multiple Linear Regression/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding Independant Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the Dummmy Variable Trap
X = X[:, 1:]

# Splitting the dataset into test and trainging sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # random state number is to test with online course

# Feature Scaling 
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test Set results
y_prediction = regressor.predict(X_test)


# BACKWARD ELIMINATION

# Import Libraries
import statsmodels.api as sm

# Appends X to an array of 1's
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_optimal =  X[:, [0, 1, 2, 3, 4, 5]] # add all featues to delete them as BWE works
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit() # Ordinary Least Squares (OLS)
regressor_OLS.summary()