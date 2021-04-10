from random import randint

import numpy as np
from sklearn.metrics import accuracy_score

import data_preparation as dp

# setting the parameters
b = 3
i = randint(0, dp.x_binarized_train.shape[0])
print("For row:{}".format(i))
if np.sum(dp.x_binarized_train[100, :]) >= b:
    print('MP Neuron inference is Malignant')
else:
    print('MP Neuron inference is Benign')

if(dp.y_train[i] == 1):
    print("Ground truth is Malignant")
else:
    print("Ground truth is Benign")

# Calculating inference using Training Dataset

b = 3
# y_pred_train = []
# acc_rows = 0
# for x, y in zip(dp.x_binarized_train, dp.y_train):
#     y_pred = (np.sum(x) >= b)
#     y_pred_train.append(y_pred)
#     acc_rows += (y == y_pred)

# print(acc_rows)
# print(acc_rows/dp.x_binarized_train.shape[0])

for b in range(dp.x_binarized_train.shape[1] + 1):
    y_pred_train = []
    acc_rows = 0
    for x, y in zip(dp.x_binarized_train, dp.y_train):
        y_pred = (np.sum(x) >= b)
        y_pred_train.append(y_pred)
        acc_rows += (y == y_pred)

    print(b)
    print(acc_rows / dp.x_binarized_train.shape[0])


# Calculating Inferenece using Testing Dataset

b = 28
y_pred_test = []
for x in dp.x_binarized_test:
    y_pred = (np.sum(x) >= b)
    y_pred_test.append(y_pred)
    acc_rows += (y == y_pred)

accuracy = accuracy_score(y_pred_test, dp.y_test)
print("Accuracy: {:.3f}".format(accuracy))
