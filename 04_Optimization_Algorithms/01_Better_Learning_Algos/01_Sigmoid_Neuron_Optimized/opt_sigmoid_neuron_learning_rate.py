"""
Implementation of Optimized Sigmoid Neuron.
    Optimizations Include:
        - Adagrad
        - RMSProp
        - Adam
"""
import argparse

import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt


class OptimizedSigNeuron:
    """
    Types of Learning Rate Optimizations:
        - ADAGRAD : Adagrad Optimizer
        - RMSProp: RMSProp Optimizer
        - Adam: Adam Optimizer
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

    def fit(self, X, Y, algo='Adagrad', epochs=100, learning_rate=0.01, beta=0.9, eps=1e-8, beta1=0.9, beta2=0.9):
        self.w_h = []
        self.b_h = []
        self.e_h = []
        self.X = X
        self.Y = Y

        if algo == 'Adagrad':
            v_w, v_b = 0, 0
            for i in tqdm(range(epochs), total=epochs, unit="epochs"):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)

                v_w += dw ** 2
                v_b += dw ** 2

                self.w -= (learning_rate / (np.sqrt(v_w) + eps)) * dw
                self.b -= (learning_rate / (np.sqrt(v_b) + eps)) * db
                self.append_log()

        elif algo == 'RMSProp':
            v_w, v_b = 0, 0
            for i in tqdm(range(epochs), total=epochs, unit="epochs"):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)

                v_w = beta * v_w + (1 - beta) * (dw ** 2)
                v_b = beta * v_b + (1 - beta) * (db ** 2)

                self.w -= (learning_rate / (np.sqrt(v_w) + eps)) * dw
                self.b -= (learning_rate / (np.sqrt(v_b) + eps)) * db
                self.append_log()

        elif algo == 'Adam':
            v_w, v_b = 0, 0
            m_w, m_b = 0, 0
            num_updates = 0
            for i in tqdm(range(epochs), total=epochs, unit="epochs"):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw = self.grad_w(x, y)
                    db = self.grad_b(x, y)
                    num_updates += 1

                    # Moment based Updates
                    m_w = (beta1 * m_w) + (1 - beta1) * dw
                    m_b = (beta1 * m_b) + (1 - beta1) * db

                    # History Based Updates
                    v_w = (beta2 * v_w) + (1 - beta2) * (dw ** 2)
                    v_b = (beta2 * v_b) + (1 - beta2) * (db ** 2)

                    # Bias Correction
                    m_w_c = m_w / (1 - np.power(beta1, num_updates))
                    m_b_c = m_b / (1 - np.power(beta1, num_updates))
                    v_w_c = v_w / (1 - np.power(beta2, num_updates))
                    v_b_c = v_b / (1 - np.power(beta2, num_updates))

                    self.w -= (learning_rate / (np.sqrt(v_w_c) + eps) * m_w_c)
                    self.b -= (learning_rate / (np.sqrt(v_b_c) + eps) * m_b_c)
                    self.append_log()

    def append_log(self):
        self.w_h.append(self.w)
        self.b_h.append(self.b)
        self.e_h.append(self.error(self.X, self.Y))


def main(args):
    X = np.asarray([3.5, 0.35, 3.2, -2.0, 1.5, -0.5])
    Y = np.asarray([0.5, 0.50, 0.5, 0.5, 0.1, 0.3])

    opt_sig_neuron = OptimizedSigNeuron(w=args.init_weight, b=args.init_bias)
    opt_sig_neuron.fit(X, Y, learning_rate=args.learning_rate,
                       algo=args.algo_type, epochs=args.epochs)

    plt.ylabel('{} Loss Surface'.format(args.algo_type))
    plt.plot(opt_sig_neuron.e_h, 'r')
    plt.plot(opt_sig_neuron.w_h, 'b')
    plt.plot(opt_sig_neuron.b_h, 'g')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_weight', type=float, help='Set initial Weight', required=True)
    parser.add_argument('--init_bias', type=float, help='Set initial bias', required=True)
    parser.add_argument('--learning_rate', type=float, help='Enter the Learning Rate[0, 1]')
    parser.add_argument('--algo_type', type=str, help='Enter the Learning Rate Optimizer: Adagrad, RMSProp, Adam')
    parser.add_argument('--epochs', type=int, help='Enter the number of Epochs the algorithm must run')
    args = parser.parse_args()
    main(args)
