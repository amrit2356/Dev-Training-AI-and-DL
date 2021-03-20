"""
Implementation of Scalar Backpropagation.(For Binary Classification)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from tqdm import tqdm
import dataset as data
import weights_visualization as visual


class FFNNScalar:
    def __init__(self):
        self.w1 = np.random.rand()
        self.w2 = np.random.rand()
        self.w3 = np.random.rand()
        self.w4 = np.random.rand()
        self.w5 = np.random.rand()
        self.w6 = np.random.rand()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

    def sigmoid(self, y):
        return (1 / (1 + np.exp(-y)))

    def forward_pass(self, x):
        # Initialization of Individual Inputs from X
        self.x1, self.x2 = x

        # Aggregation & Activation on 1st Neuron
        self.a1 = (self.x1 * self.w1 + self.x2 * self.w2) + self.b1
        self.h1 = self. sigmoid(self.a1)

        # Aggregation & Activation on 2nd Neuron
        self.a2 = (self.x1 * self.w3 + self.x2 * self.w4) + self.b2
        self.h2 = (self.sigmoid(self.a2))

        # Aggregation & Activation on 3rd Neuron
        self.a3 = (self.h1 * self.w5 + self.h2 * self.w6) + self.b3
        self.h3 = (self.sigmoid(self.a3))
        return self.h3

    def grad(self, x, y):
        self.forward_pass(x)

        self.dw5 = (self.h3 - y) * ((1 - self.h3) * self.h3) * self.h1
        self.dw6 = (self.h3 - y) * ((1 - self.h3) * self.h3) * self.h2
        self.db3 = (self.h3 - y) * ((1 - self.h3) * self.h3)

        self.dw1 = (self.h3 - y) * ((1 - self.h3) * self.h3) * \
            self.w5 * ((1 - self.h1) * self.h1) * self.x1
        self.dw2 = (self.h3 - y) * ((1 - self.h3) * self.h3) * \
            self.w5 * ((1 - self.h1) * self.h1) * self.x2
        self.db1 = (self.h3 - y) * ((1 - self.h3) * self.h3) * \
            self.w5 * ((1 - self.h1) * self.h1)

        self.dw3 = (self.h3 - y) * ((1 - self.h3) * self.h3) * \
            self.w6 * ((1 - self.h2) * self.h2) * self.x1
        self.dw4 = (self.h3 - y) * ((1 - self.h3) * self.h3) * \
            self.w6 * ((1 - self.h2) * self.h2) * self.x2
        self.db2 = (self.h3 - y) * ((1 - self.h3) * self.h3) * \
            self.w6 * ((1 - self.h2) * self.h2)

    def fit(self, X, Y, epochs=100, learning_rate=0.001, initialize=True, display_loss=False, display_weight=False, loss_type='ce'):

        if initialize:
            np.random.seed(0)
            self.w1 = np.random.randn()
            self.w2 = np.random.randn()
            self.w3 = np.random.randn()
            self.w4 = np.random.randn()
            self.w5 = np.random.randn()
            self.w6 = np.random.randn()
            self.b1 = 0
            self.b2 = 0
            self.b3 = 0

        if display_loss:
            loss = {}

        for i in tqdm(range(epochs), total=epochs, unit="epoch"):
            dw1, dw2, dw3, dw4, dw5, dw6, db1, db2, db3 = [0] * 9
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3

            m = X.shape[0]
            self.w1 -= learning_rate * dw1 / m
            self.w2 -= learning_rate * dw2 / m
            self.w3 -= learning_rate * dw3 / m
            self.w4 -= learning_rate * dw4 / m
            self.w5 -= learning_rate * dw5 / m
            self.w6 -= learning_rate * dw6 / m
            self.b1 -= learning_rate * db1 / m
            self.b2 -= learning_rate * db2 / m
            self.b3 -= learning_rate * db3 / m

            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = mean_squared_error(Y, Y_pred)

            if display_weight:
                weight_matrices = []
                weight_matrix = np.array([[0, self.b3, self.w5, self.w6, 0, 0], [
                                         self.b1, self.w1, self.w2, self.b2, self.w3, self.w4]])
                weight_matrices.append(weight_matrix)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def predict_h1(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(self.h1)
        return np.array(Y_pred)

    def predict_h2(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(self.h2)
        return np.array(Y_pred)

    def predict_h3(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(self.h1)
        return np.array(Y_pred)

    def plot_boundary(self):
        xx, yy = visual.make_meshgrid(data.X_train[:, 0], data.X_train[:, 1])
        predict_functions = [self.predict_h1, self.predict_h2, self.predict_h3]

        for i in range(3):
            fig, ax = plt.subplots(figsize=(10, 5))
            visual.plot_contours(ax, predict_functions[i], xx, yy, cmap=data.my_cmap, alpha=0.2)
            ax.scatter(data.X_train[:, 0], data.X_train[:, 1], c=data.Y_train, cmap=data.my_cmap, alpha=0.8)
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title("h" + str(i + 1))

        return True

def main():
    plt.scatter(data.data[:, 0], data.data[:, 1], c=data.labels, cmap=data.my_cmap)
    plt.show()

    ffn = FFNNScalar()
    ffn.fit(data.X_train, data.Y_train, epochs=2000,
            learning_rate=5, display_loss=True, display_weight=True)

    Y_pred_train = ffn.predict(data.X_train)
    Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

    Y_pred_val = ffn.predict(data.X_val)
    Y_pred_binarised_val = (Y_pred_val >= 0.5).astype("int").ravel()

    accuracy_train = accuracy_score(Y_pred_binarised_train, data.Y_train)
    accuracy_val = accuracy_score(Y_pred_binarised_val, data.Y_val)

    print("Training accuracy", round(accuracy_train, 2))
    print("Validation accuracy", round(accuracy_val, 2))

    plt.scatter(data.X_train[:, 0], data.X_train[:, 1], c=Y_pred_binarised_train,
                cmap=data.my_cmap, s=15 * (np.abs(Y_pred_binarised_train - data.Y_train) + .2))
    plt.show()

    ffn.plot_boundary()


if __name__ == "__main__":
    main()
