"""
Implementation  of Error Space of the Sigmoid Neuron
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from data_prep_class import DataPreparation

class Sigmoid():
    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self, X):
        return (np.dot(X, self.w.T) + self.b)

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def grad_w(self, y_pred, y, x):
        return (y_pred - y) * (1 - y_pred) * (y_pred) * x

    def grad_b(self, y_pred, y):
        return (y_pred - y) * (1 - y_pred) * (y_pred) * 1

    def loss_mse(self, y, y_pred):
        return mean_squared_error(y_pred, y)

    def fit(self, X, Y, learning_rate=0.001, epochs=100, display_loss=False):
        self.w = np.random.rand(1, X.shape[1])
        self.b = 0
        if display_loss:
            loss = {}

        for i in range(epochs):
            dw = 0
            db = 0
            for x, y in zip(X, Y):
                y_pred = self.sigmoid(self.perceptron(x))
                dw += self.grad_w(y_pred, y, x)
                db += self.grad_b(y_pred, y)
            self.w -= dw * learning_rate
            self.b -= db * learning_rate

            if display_loss:
                Y_pred = self.sigmoid(self.perceptron(X))
                loss[i] = self.loss_mse(Y_pred, Y)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Loss Value')
            plt.show()

    def predict(self, X):
        y_pred = []
        for x in X:
            y = self.sigmoid(self.perceptron(x))
            y_pred.append(y)


def main(args):
    # Creation of Dataset Object for Dataset Prep.
    dataset_obj = DataPreparation()
    data = dataset_obj.readcsv(args.dataset_path)

    # Generate Binary Threshold
    X, Y, threshold, Y_binarised = dataset_obj.generate_binary_threshold(data, 4.2)

    # Splitting the Dataset
    X_Train, X_Test, Y_Train, Y_Test = dataset_obj.dataset_split(X, Y, stratify=Y_binarised)

    # Standardization of Dataset using StandardScaler and MinMax Scaler
    x_sca_tn, x_sca_tes, y_sca_tn, y_sca_tes, sca_thresh = dataset_obj.standardization(
        X_train=X_Train, X_test=X_Test, Y_train=Y_Train, Y_test=Y_Test, threshold=threshold)

    # Generation of Binarized Values for Y Scaled Train and Test data.
    Y_binarised_train, Y_binarised_test = dataset_obj.threshold_scaling(y_sca_tn, y_sca_tes, sca_thresh)

    # Creation of Sigmoid Neuron Class
    sn = Sigmoid()

    # Fitting the Scaled X_train and X_test values on the Sigmoid Neuron
    sn.fit(x_sca_tn, y_sca_tn, learning_rate=0.015, epochs=2000, display_loss=True)

    y_pred_train = sn.predict(y_sca_tn)
    y_pred_test = sn.predict(y_sca_tes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Enter Dataset File Path', required=True)
    args = parser.parse_args()
    main(args)
