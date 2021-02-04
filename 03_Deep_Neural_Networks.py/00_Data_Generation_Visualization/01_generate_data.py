import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import train_test_split


# Generate a random seed
np.random.seed(0)
# Contour Map Initialization
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])

# Generate Data using make_blobs()
data, labels = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=0)
print(data.shape, labels.shape)

# Plot the data generated and visualize its distributions
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

# modifying the dataset to have 2 centers
labels_orig = labels
labels = np.mod(labels_orig, 2)

# Visualizing the same dataset but only 2 classes
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

# Splitting the dataset into training and testing Dataset
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
print(X_train.shape, X_val.shape)
