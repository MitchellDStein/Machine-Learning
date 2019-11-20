# Data Preprocessing

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('Data.csv')

# create matrix of features
X = dataset.iloc[:, :-1].values

# Create dependant variable vector
y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])


# Encoding categorical data
#===========================
# Encoding independent variable (countries)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding Y data (purchased)
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

