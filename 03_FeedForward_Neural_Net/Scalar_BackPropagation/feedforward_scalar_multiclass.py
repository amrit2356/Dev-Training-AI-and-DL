"""
Implementation of Scalar Backpropagation.(For Multiclass Classification)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import dataset

class FFNMulticlass:
    def __init__(self):
        np.random.seed(0)
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.w7 = np.random.randn()
        self.w8 = np.random.randn()
        self.w9 = np.random.randn()
        self.w10 = np.random.randn()
        self.w11 = np.random.randn()
        self.w12 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0

    def sigmoid(self, y):
        return (1 / (1 + np.exp(-y)))

    def forward_pass(self, x):
        self.x1, self.x2 = x

        # First Layer
        self.a1 = (self.w1 * self.x1 + self.w2 * self.x2) + self.b1
        self.h1 = self.sigmoid(self.a1)

        self.a2 = (self.w3 * self.x1 + self.w4 * self.x2) + self.b2
        self.h2 = self.sigmoid(self.a2)

        # 2nd Layer
        self.a3 = (self.w5 * self.h1 + self.w6 * self.h2) + self.b3
        self.a4 = (self.w7 * self.h1 + self.w8 * self.h2) + self.b4
        self.a5 = (self.w9 * self.h1 + self.w10 * self.h2) + self.b5
        self.a6 = (self.w11 * self.h1 + self.w12 * self.h2) + self.b6

        # softmax Function
        sum_exps = np.sum([np.exp(self.a3), np.exp(self.a4),
                           np.exp(self.a5), np.exp(self.a6)])
        self.h3 = np.exp(self.a3) / sum_exps
        self.h4 = np.exp(self.a4) / sum_exps
        self.h5 = np.exp(self.a5) / sum_exps
        self.h6 = np.exp(self.a6) / sum_exps

        return np.array([self.h3, self.h4, self.h5, self.h6])

    def backward_pass(self, x, y):
        self.forward_pass(x)
        self.y1, self.y2, self.y3, self.y4 = y
        # Output Layer Gradients wrt w, b
        self.dw5 = (self.h3 - self.y1) * self.h1
        self.dw6 = (self.h3 - self.y1) * self.h2
        self.db3 = (self.h3 - self.y1)

        self.dw7 = (self.h4 - self.y2) * self.h1
        self.dw8 = (self.h4 - self.y2) * self.h2
        self.db4 = (self.h4 - self.y2)

        self.dw9 = (self.h5 - self.y3) * self.h1
        self.dw10 = (self.h5 - self.y3) * self.h2
        self.db5 = (self.h5 - self.y3)

        self.dw11 = (self.h6 - self.y4) * self.h1
        self.dw12 = (self.h6 - self.y4) * self.h2
        self.db6 = (self.h6 - self.y4)

        # Hidden Layer and calculating w1, w2 and b1 using h1(chain rule)
        self.dh1 = (self.h3 - self.y1) * self.w5 + (self.h4 - self.y2) * \
            self.w7 + (self.h5 - self.y3) * self.w9 + \
            (self.h6 - self.y4) * self.w11
        self.dw1 = self.dh1 * self.h1 * (1 - self.h1) * self.x1
        self.dw2 = self.dh1 * self.h1 * (1 - self.h1) * self.x2
        self.db1 = self.dh1 * self.h1 * (1 - self.h1)

        self.dh2 = (self.h3 - self.y1) * self.w6 + (self.h4 - self.y2) * self.w8 + \
            (self.h5 - self.y3) * self.w10 + (self.h6 - self.y4) * self.w12
        self.dw3 = self.dh2 * self.h2 * (1 - self.h2) * self.x1
        self.dw4 = self.dh2 * self.h2 * (1 - self.h2) * self.x2
        self.db2 = self.dh2 * self.h2 * (1 - self.h2)

    def backward_pass_short(self, x, y):
        self.forward_pass(x)
        self.y1, self.y2, self.y3, self.y4 = y

        self.da3 = (self.h3 - self.y1)
        self.da4 = (self.h4 - self.y2)
        self.da5 = (self.h5 - self.y3)
        self.da6 = (self.h6 - self.y4)

        self.dw5 = self.da3 * self.h1
        self.dw6 = self.da3 * self.h2
        self.db3 = self.da3

        self.dw7 = self.da4 * self.h1
        self.dw8 = self.da4 * self.h2
        self.db4 = self.da4

        self.dw9 = self.da5 * self.h1
        self.dw10 = self.da5 * self.h2
        self.db5 = self.da5

        self.dw11 = self.da6 * self.h1
        self.dw12 = self.da6 * self.h2
        self.db6 = self.da6

        self.dh1 = self.da3 * self.w5 + self.da4 * \
            self.w7 + self.da5 * self.w9 + self.da6 * self.w11

        self.dh2 = self.da3 * self.w6 + self.da4 * self.w8 + \
            self.da5 * self.w10 + self.da6 * self.w12

        self.da1 = self.dh1 * self.h1 * (1 - self.h1)
        self.da2 = self.dh2 * self.h2 * (1 - self.h2)

        self.dw1 = self.da1 * self.x1
        self.dw2 = self.da1 * self.x2
        self.db1 = self.da1

        self.dw3 = self.da2 * self.x1
        self.dw4 = self.da2 * self.x2
        self.db2 = self.da2

    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False, display_weight=False):

        if display_loss:
            loss = {}

        for i in tqdm(range(epochs), total=epochs, unit="epoch"):
            dw1, dw2, dw3, dw4, dw5, dw6, dw7, dw8, dw9, dw10, dw11, dw12, db1, db2, db3, db4, db5, db6 = [
                0] * 18
            for x, y in zip(X, Y):
                self.backward_pass_short(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                dw7 += self.dw7
                dw8 += self.dw8
                dw9 += self.dw9
                dw10 += self.dw10
                dw11 += self.dw11
                dw12 += self.dw12
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3
                db1 += self.db4
                db2 += self.db5
                db3 += self.db6

            m = X.shape[0]
            self.w1 -= learning_rate * dw1 / m
            self.w2 -= learning_rate * dw2 / m
            self.w3 -= learning_rate * dw3 / m
            self.w4 -= learning_rate * dw4 / m
            self.w5 -= learning_rate * dw5 / m
            self.w6 -= learning_rate * dw6 / m
            self.w7 -= learning_rate * dw7 / m
            self.w8 -= learning_rate * dw8 / m
            self.w9 -= learning_rate * dw9 / m
            self.w10 -= learning_rate * dw10 / m
            self.w11 -= learning_rate * dw11 / m
            self.w12 -= learning_rate * dw12 / m
            self.b1 -= learning_rate * db1 / m
            self.b2 -= learning_rate * db2 / m
            self.b3 -= learning_rate * db3 / m
            self.b4 -= learning_rate * db4 / m
            self.b5 -= learning_rate * db5 / m
            self.b6 -= learning_rate * db6 / m

            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)

            if display_weight:
                weight_matrices = []
                weight_matrix = np.array([[self.b3, self.w5, self.w6,
                                           self.b4, self.w7, self.w8,
                                           self.b5, self.w9, self.w10,
                                           self.b6, self.w11, self.w12],
                                          [0, 0, 0,
                                           self.b1, self.w1, self.w2,
                                           self.b2, self.w3, self.w4,
                                           0, 0, 0]])
                weight_matrices.append(weight_matrix)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

def main():
    X_train, X_val, Y_train, Y_val = train_test_split(dataset.data, dataset.labels_orig, stratify=dataset.labels_orig, random_state=0)
    print(X_train.shape, X_val.shape, dataset.labels_orig.shape)

    enc = OneHotEncoder()
    # 0 -> (1, 0, 0, 0), 1 -> (0, 1, 0, 0), 2 -> (0, 0, 1, 0), 3 -> (0, 0, 0, 1)
    y_OH_train = enc.fit_transform(np.expand_dims(Y_train, 1)).toarray()
    y_OH_val = enc.fit_transform(np.expand_dims(Y_val, 1)).toarray()
    print(y_OH_train.shape, y_OH_val.shape)

    ffsn_multi_specific = FFNMulticlass()
    ffsn_multi_specific.fit(X_train, y_OH_train, epochs=2000, learning_rate=1, display_loss=True, display_weight=True)
    Y_pred_train = ffsn_multi_specific.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train, 1)

    Y_pred_val = ffsn_multi_specific.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val, 1)

    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)

    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))

    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_pred_train, cmap=dataset.my_cmap, s=15 * (np.abs(np.sign(Y_pred_train - Y_train)) + .1))
    plt.show()

if __name__ == "__main__":
    main()
