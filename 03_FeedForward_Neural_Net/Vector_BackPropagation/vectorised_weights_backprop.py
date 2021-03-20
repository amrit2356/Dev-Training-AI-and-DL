"""
Feedforward Neural Networks: Weights Vectorized
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import log_loss


class FF_MultiClass_WeightVectorised:

    def __init__(self, W1, W2):
        self.W1 = W1.copy()
        self.W2 = W2.copy()
        self.B1 = np.zeros((1, 2))
        self.B2 = np.zeros((1, 4))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def forward_pass(self, x):
        x = x.reshape(1, -1)  # (1, 2)
        self.A1 = np.matmul(x, self.W1) + self.B1  # (1, 2) * (2, 2) -> (1, 2)
        self.H1 = self.sigmoid(self.A1)  # (1, 2)
        self.A2 = np.matmul(self.H1, self.W2) + \
            self.B2  # (1, 2) * (2, 4) -> (1, 4)
        self.H2 = self.softmax(self.A2)  # (1, 4)
        return self.H2

    def grad_sigmoid(self, x):
        return x * (1 - x)

    def grad(self, x, y):
        self.forward_pass(x)
        x = x.reshape(1, -1)  # (1, 2)
        y = y.reshape(1, -1)  # (1, 4)

        self.dA2 = self.H2 - y  # (1, 4)

        self.dW2 = np.matmul(self.H1.T, self.dA2)  # (2, 1) * (1, 4) -> (2, 4)
        self.dB2 = self.dA2  # (1, 4)
        self.dH1 = np.matmul(self.dA2, self.W2.T)  # (1, 4) * (4, 2) -> (1, 2)
        self.dA1 = np.multiply(
            self.dH1, self.grad_sigmoid(self.H1))  # -> (1, 2)

        self.dW1 = np.matmul(x.T, self.dA1)  # (2, 1) * (1, 2) -> (2, 2)
        self.dB1 = self.dA1  # (1, 2)

    def fit(self, X, Y, epochs=1, learning_rate=1, display_loss=False):

        if display_loss:
            loss = {}

        for i in tqdm(range(epochs), total=epochs, unit="epoch"):
            dW1 = np.zeros((2, 2))
            dW2 = np.zeros((2, 4))
            dB1 = np.zeros((1, 2))
            dB2 = np.zeros((1, 4))
            for x, y in zip(X, Y):
                self.grad(x, y)
                dW1 += self.dW1
                dW2 += self.dW2
                dB1 += self.dB1
                dB2 += self.dB2

            m = X.shape[0]
            self.W2 -= learning_rate * (dW2 / m)
            self.B2 -= learning_rate * (dB2 / m)
            self.W1 -= learning_rate * (dW1 / m)
            self.B1 -= learning_rate * (dB1 / m)

            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = log_loss(np.argmax(Y, axis=1), Y_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            plt.savefig(
                'Dev-Training-DL/Exercises/02_FeedForward_Neural_Net/Vector_BackPropagation/Loss_Graphs/Vectorised_Weights_Backprop.png')
            plt.show()

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()


def main():
    pass
