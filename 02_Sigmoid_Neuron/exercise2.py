"""
Implementation  of Sigmoid Neuron with Mean Square Error Loss
and Cross Entropy Loss
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm
from data_prep_class import DataPreparation


class Sigmoid():
    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self, X):
        return (np.dot(X, self.w.T) + self.b)

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def grad_w(self, y_pred, y, x, loss_type):
        if loss_type == 'ce':
            return (y_pred - y) * x
        else:
            return (y_pred - y) * (1 - y_pred) * (y_pred) * x

    def grad_b(self, y_pred, y, loss_type):
        if loss_type == 'ce':
            return (y_pred - y)
        else:
            return (y_pred - y) * (1 - y_pred) * (y_pred) * 1

    def loss_mse(self, y_pred, y):
        return mean_squared_error(y_pred, y)

    def loss_ce(self, y_pred, y):
        err = 0.0
        err += -((1 - y) * math.log2(1 - y_pred) + y * math.log2(y_pred))
        return err

    def fit(self, X, Y, learning_rate=0.001, epochs=100, display_loss=False, loss_type='mse'):
        self.w = np.random.rand(1, X.shape[1])
        self.b = 0
        if display_loss:
            loss = {}

        for i in tqdm(range(epochs), total=epochs, unit="epoch"):
            dw = 0
            db = 0
            for x, y in zip(X, Y):
                # Calcuation of Predicted y values
                y_pred = self.sigmoid(self.perceptron(x))
                dw += self.grad_w(y_pred, y, x, loss_type)
                db += self.grad_b(y_pred, y, loss_type)

            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if display_loss:
                Y_pred = self.sigmoid(self.perceptron(X))
                if loss_type == 'mse':
                    loss[i] = self.loss_mse(Y_pred, Y)
                if loss_type == 'ce':
                    loss[i] = self.loss_ce(y_pred, y)

        if display_loss:
            plt.plot(loss.values())
            if loss_type == 'mse':
                plt.ylabel(' Mean Square Error Loss Value')
                plt.xlabel('Epochs')
                plt.savefig('Dev-Training-DL/Exercises/01_Sigmoid_Neuron/Loss_Graphs/MSE_loss_graph.png')
            if loss_type == 'ce':
                plt.ylabel(' Cross Entropy Error Loss Value')
                plt.xlabel('Epochs')
                plt.savefig('Dev-Training-DL/Exercises/01_Sigmoid_Neuron/Loss_Graphs/CE_loss_graph.png')
            plt.show()

    def predict(self, X):
        y_pred = []
        for x in X:
            y = self.sigmoid(self.perceptron(x))
            y_pred.append(y)
        return np.array(y_pred)


def main(args):
    # Creation of Dataset Object for Dataset Prep.
    dataset_obj = DataPreparation()
    data = dataset_obj.readcsv(args.dataset_path)

    # Generate Binary Threshold
    X, Y, threshold, Y_binarised = dataset_obj.generate_binary_threshold(
        data, 4.2)

    # Splitting the Dataset
    X_Train, X_Test, Y_Train, Y_Test = dataset_obj.dataset_split(
        X, Y, stratify=Y_binarised)

    # Standardization of Dataset using StandardScaler and MinMax Scaler
    x_sca_tn, x_sca_tes, y_sca_tn, y_sca_tes, sca_thresh = dataset_obj.standardization(
        X_train=X_Train, X_test=X_Test, Y_train=Y_Train, Y_test=Y_Test, threshold=threshold)

    # Generation of Binarized Values for Y Scaled Train and Test data.
    Y_binarised_train, Y_binarised_test = dataset_obj.threshold_scaling(
        y_sca_tn, y_sca_tes, sca_thresh)

    # Creation of Sigmoid Neuron Class
    sn = Sigmoid()

    # Fitting the Scaled X_train and X_test values on the Sigmoid Neuron
    sn.fit(x_sca_tn, y_sca_tn, learning_rate=args.learning_rate,
           epochs=args.epochs, display_loss=True, loss_type=args.loss_type)

    y_pred_train = sn.predict(x_sca_tn)
    y_pred_test = sn.predict(x_sca_tes)

    # Binarizing the Predicted values of Y for train and test set.
    y_pred_binarised_train = (y_pred_train > sca_thresh).astype("int").ravel()
    y_pred_binarised_test = (y_pred_test > sca_thresh).astype("int").ravel()

    # Measuring Training and Testing Accuracy
    acc_train = accuracy_score(y_pred_binarised_train, Y_binarised_train)
    acc_test = accuracy_score(y_pred_binarised_test, Y_binarised_test)
    print("Training Accuracy: {:.3f}".format(acc_train))
    print("Testing Accuracy: {:.3f}".format(acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Enter Dataset File Path', required=True)
    parser.add_argument('--learning_rate', type=float,
                        help='Enter the Learning Rate[0, 1]')
    parser.add_argument('--epochs', type=int,
                        help='Enter the Number of Epochs')
    parser.add_argument('--display_loss', type=bool,
                        help='Set Display Loss Graph: To view loss graph')
    parser.add_argument('--loss_type', type=str,
                        help='Choose Loss Type: MSE or CE')
    args = parser.parse_args()
    main(args)
