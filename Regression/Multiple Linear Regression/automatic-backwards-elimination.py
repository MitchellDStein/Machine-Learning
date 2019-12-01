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


# AUTOMATIC Backwards Elimination
# Import Libraries
import statsmodels.api as sm
def backwardElimination(x_optimal, sl):
    numVars = len(x_optimal[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x_optimal).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x_optimal = np.delete(x_optimal, j, 1)
    regressor_OLS.summary()
    return x_optimal

# significance value to test p-values against
SL = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4]]
X_Modeled = backwardElimination(X_optimal, SL)
