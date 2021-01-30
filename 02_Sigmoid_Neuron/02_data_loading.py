import numpy as np
import pandas as pd

# Loading the data and preparing the dataset
data = pd.read_csv('02_Sigmoid_Neuron/data/mobile_cleaned.csv')
print(data.head())
print(data.shape)
X = data.drop('Rating', axis=1)
Y = data['Rating'].values
print(Y)

# adding Threshold for Binary Classification.
threshold = 4.2
data['Class'] = (data['Rating'] >= threshold).astype(np.int)
Y_binarised = data['Class'].values
