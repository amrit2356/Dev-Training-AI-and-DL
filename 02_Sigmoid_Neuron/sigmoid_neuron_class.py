import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from data_prep_class import DataPreparation
# Class for Sigmoid


class SigmoidNeuron:
    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self, x):
        return np.dot(x, self.w.T) + self.b

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):
        # initialise w, b
        if initialise:
            self.w = np.random.randn(1, X.shape[1])
            self.b = 0

        if display_loss:
            loss = {}

        for i in range(epochs):
            dw = 0
            db = 0
            for x, y in zip(X, Y):
                dw += self.grad_w(x, y)
                db += self.grad_b(x, y)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

            if display_loss:
                Y_pred = self.sigmoid(self.perceptron(X))
                loss[i] = mean_squared_error(Y_pred, Y)
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.sigmoid(self.perceptron(x))
            Y_pred.append(y_pred)
        return np.array(Y_pred)


def main():
    # Step 1: Data Preparation
    # Creation of Data Prep Object
    dataset_obj = DataPreparation()

    # Loading of Dataset
    data = dataset_obj.readcsv('02_Sigmoid_Neuron/data/mobile_cleaned.csv')

    # Generation of Binary threshold value.
    X, Y, threshold, Y_binarised = dataset_obj.generate_binary_threshold(
        data, 4.2)

    # Splitting the dataset
    X_train, X_test, Y_train, Y_test = dataset_obj.dataset_split(
        X, Y, stratify=Y_binarised, random_state=0)

    # Standardization using StandardScaler and MinMax Scaler
    x_sca_tn, x_sca_tes, y_sca_tn, y_sca_tes, sca_thresh = dataset_obj.standardization(
        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, threshold=threshold)
    print(sca_thresh)

    # Generation of Binarized Values for Y Scaled Train and Test data.
    Y_binarised_train, Y_binarised_test = dataset_obj.threshold_scaling(y_sca_tn, y_sca_tes, sca_thresh)
    # Step 2: Training the Sigmoid Neuron
    sn = SigmoidNeuron()
    sn.fit(x_sca_tn, y_sca_tn, epochs=2000, learning_rate=0.015, display_loss=True)

    Y_pred_train = sn.predict(x_sca_tn)
    Y_pred_test = sn.predict(x_sca_tes)

    Y_pred_binarised_train = (Y_pred_train > sca_thresh).astype("int").ravel()
    Y_pred_binarised_test = (Y_pred_test > sca_thresh).astype("int").ravel()

    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_binarised_train)
    accuracy_test = accuracy_score(Y_pred_binarised_test, Y_binarised_test)
    print(accuracy_train, accuracy_test)


if __name__ == "__main__":
    main()
