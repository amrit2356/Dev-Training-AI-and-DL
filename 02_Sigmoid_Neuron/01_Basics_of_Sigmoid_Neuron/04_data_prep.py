import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Loading the data and preparing the dataset
data = pd.read_csv('02_Sigmoid_Neuron/data/mobile_cleaned.csv')
print(data.head())
print(data.shape)
X = data.drop('Rating', axis=1)
print(X)
Y = data['Rating'].values
print(Y)

# Adding Threshold for Binary Classification.
threshold = 4.2
data['Class'] = (data['Rating'] >= threshold).astype(np.int)
print(data['Class'].value_counts(normalize=True))
Y_binarised = data['Class'].values

# Preprocessing the Dataset
# Splitting the Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=0, stratify=Y_binarised)
print(X_train.shape, X_test.shape)

# Using Standard Scaler Function
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

# Using MinMax Scaler Function
minmax_scaler = MinMaxScaler()
Y_scaled_train = minmax_scaler.fit_transform(Y_train.reshape(-1, 1))

print(np.min(Y_scaled_train))

Y_scaled_test = minmax_scaler.transform(Y_test.reshape(-1, 1))
scaled_threshold = list(minmax_scaler.transform(
    np.array([threshold]).reshape(1, -1)))[0][0]
print(scaled_threshold)

Y_binarised_train = (Y_scaled_train > scaled_threshold).astype("int").ravel()
Y_binarised_test = (Y_scaled_test > scaled_threshold).astype("int").ravel()
