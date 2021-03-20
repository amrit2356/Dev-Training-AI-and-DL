"""
Implementation of Optimized Sigmoid Neuron.
    Optimizations Include:
        - Basic GD
        - Moment Based GD
        - Nesterov Acc. GD
        - Mini-Batch
        - Stochastic GD
Note: GD -> Gradient Descent
"""
import argparse

import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt


class OptimizedSigNeuron:
    """
    Types of Optimizations:
        - GD : Gradient Descent
        - MGD: Moment based Gradient Descent
        - NAGD: Nesterov Accelrated Gradient Descent
        - MiniBatch: Minibatch Gradient Descent
        - SGD: Stochastic Gradient Descent
    """

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.w_h = []
        self.b_h = []
        self.e_h = []

    def sigmoid(self, x, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return 1. / (1. + np.exp(-(w * x + b)))

    def error(self, X, Y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        err = 0
        for x, y in zip(X, Y):
            err += 0.5 * (self.sigmoid(x, w, b) - y) ** 2
        return err

    def grad_w(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred)

    def fit(self, X, Y, algo='GD', epochs=100, learning_rate=0.01, gamma=0.9, minibatch_size=100):
        self.w_h = []
        self.b_h = []
        self.e_h = []
        self.X = X
        self.Y = Y

        if algo == 'GD':
            for i in tqdm(range(epochs), total=epochs, unit='epochs'):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)

                self.w = self.w - learning_rate * dw
                self.b = self.w - learning_rate * db
                self.append_log()

        elif algo == 'MGD':
            v_w, v_b = 0, 0
            for i in tqdm(range(epochs), total=epochs, unit='epochs'):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                v_w = gamma * v_w + learning_rate * dw
                v_b = gamma * v_b + learning_rate * db
                self.w -= v_w
                self.b -= v_b
                self.append_log()

        elif algo == 'NAGD':
            v_w, v_b = 0, 0
            for i in tqdm(range(epochs), total=epochs, unit='epochs'):
                self.w = self.w - gamma * v_w
                self.b = self.b - gamma * v_b
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y, self.w - v_w, self.b - v_b)
                    db += self.grad_b(x, y, self.w - v_w, self.b - v_b)
                self.w -= learning_rate * dw
                self.b -= learning_rate * db
                v_w = gamma * v_w + learning_rate * dw
                v_b = gamma * v_b + learning_rate * db
                self.append_log()

        elif algo == 'SGD':
            for i in tqdm(range(epochs), total=epochs, unit='epochs'):
                db, dw = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                    self.w -= learning_rate * dw
                    self.b -= learning_rate * db
                    self.append_log()

        elif algo == 'Minibatch':
            for i in tqdm(range(epochs), total=epochs, unit='epochs'):
                db, dw = 0, 0
                points_seen = 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                    points_seen += 1
                    if minibatch_size % points_seen == 0:
                        self.w -= (learning_rate * dw) / minibatch_size
                        self.b -= (learning_rate * db) / minibatch_size
                        self.append_log()
                        dw, db = 0, 0

    def append_log(self):
        self.w_h.append(self.w)
        self.b_h.append(self.b)
        self.e_h.append(self.error(self.X, self.Y))


def main(args):
    X = np.asarray([3.5, 0.35, 3.2, -2.0, 1.5, -0.5])
    Y = np.asarray([0.5, 0.50, 0.5, 0.5, 0.1, 0.3])

    opt_sig_neuron = OptimizedSigNeuron(w=args.init_weight, b=args.init_bias)
    opt_sig_neuron.fit(X, Y, learning_rate=args.learning_rate,
                       algo=args.algo_type, epochs=args.epochs, minibatch_size=args.minibatch_size)

    plt.plot(opt_sig_neuron.e_h, 'r')
    plt.plot(opt_sig_neuron.w_h, 'b')
    plt.plot(opt_sig_neuron.b_h, 'g')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_weight', type=float, help='Set initial Weight', required=True)
    parser.add_argument('--init_bias', type=float, help='Set initial bias', required=True)
    parser.add_argument('--learning_rate', type=float, help='Enter the Learning Rate[0, 1]')
    parser.add_argument('--algo_type', type=str, help='Enter the Algorithm: GD, MGD, NAGD, Minibatch', required=True)
    parser.add_argument('--epochs', type=int, help='Enter the number of Epochs the algorithm must run')
    parser.add_argument('--minibatch_size', type=int, help='Enter Minibatch-size')
    args = parser.parse_args()
    main(args)
