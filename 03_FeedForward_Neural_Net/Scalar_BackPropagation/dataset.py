"""
Creation of Random Dataset using make_blobs and
plotting them.
"""
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


# User-Defined Contour Maps
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "yellow", "green"])


# Generate data using make_blobs function
data, labels = make_blobs(n_samples=1000, n_features=2,
                          centers=4, random_state=0)
print(data.shape, labels.shape)

# Plotting the Dataset on the Graph
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

labels_orig = labels
labels = np.mod(labels_orig, 2)

plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=my_cmap)
plt.show()

X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
print(X_train.shape, X_val.shape)
