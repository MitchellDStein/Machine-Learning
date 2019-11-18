# Data Preprocessing

# importing libraries
import numpy as numpy
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv('Data.csv')

# create matrix of features
X = dataset.iloc[:, :-1].values

# Create dependant variable vector
y = dataset.iloc[:, 3].values