"""
Generation of Random Dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Generation of Colour Maps
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

# Set Seed Value
np.random.seed(0)

# Generate Random Data
data, labels = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)
print(data.shape, labels.shape)

# Visualization of Random Dataset
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

# Splitting of dataset in Training and Testing Dataset.
x_train, x_val, y_train, y_val = train_test_split(data, labels, stratify=labels, random_state=0)
print(x_train.shape, x_val.shape, labels.shape)

# one hot encoding of the dataset.
enc = OneHotEncoder()
# 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
y_OH_train = enc.fit_transform(np.expand_dims(y_train, 1)).toarray()
y_OH_val = enc.fit_transform(np.expand_dims(y_val, 1)).toarray()
print(y_OH_train.shape, y_OH_val.shape)


class DataPreparation:
    def __init__(self):
        self.data = None
        self.labels = None
        self.x_train = None
        self.x_val = None
        self.y_train = None
        self.y_val = None
        self.my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

    def generate_random_dataset(self, n_samples=1000, n_features=2, centers=4, random_state=0):
        self.data, self.labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
        return self.data, self.labels

    def split_dataset(self, data, labels, stratify=labels, random_state=0):
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(data, labels, stratify=stratify, random_state=random_state)
        return self.x_train, self.x_val, self.y_train, self.y_val
