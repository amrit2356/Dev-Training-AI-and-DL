"""
Implementation of Optimized Strategies for FeedForward Neural Network
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
import dataset

class FFNetwork:

    def __init__(self, W1, W2):

        self.params = {}
        self.params["W1"] = W1.copy()
        self.params["W2"] = W2.copy()
        self.params["B1"] = np.zeros((1, 2))
        self.params["B2"] = np.zeros((1, 4))
        self.num_layers = 2
        self.gradients = {}
        self.update_params = {}
        self.prev_update_params = {}
        for i in range(1, self.num_layers + 1):
            self.update_params["v_w" + str(i)] = 0
            self.update_params["v_b" + str(i)] = 0
            self.update_params["m_b" + str(i)] = 0
            self.update_params["m_w" + str(i)] = 0
            self.prev_update_params["v_w" + str(i)] = 0
            self.prev_update_params["v_b" + str(i)] = 0
        self.path = 'Dev-Training-DL/Exercises/03_Optimization_Algorithms/01_Better_Learning_Algos/02_FeedForward_NN_Optimized/Loss_Graphs/'

    def forward_activation(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def grad_activation(self, X):
        return X * (1 - X)

    def softmax(self, X):
        exps = np.exp(X)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def forward_pass(self, X, params=None):
        if params is None:
            params = self.params
        self.A1 = np.matmul(X, params["W1"]) + params["B1"]  # (N, 2) * (2, 2) -> (N, 2)
        self.H1 = self.forward_activation(self.A1)  # (N, 2) * (2, 4) -> (N, 4)
        self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"]
        self.H2 = self.softmax(self.A2)  # (N, 4)
        return self.H2

    def grad(self, X, Y, params=None):
        if params is None:
            params = self.params

        self.forward_pass(X, params)
        m = X.shape[0]
        self.gradients["dA2"] = self.H2 - Y  # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"])  # (2, N) * (N, 4) -> (2, 4)
        self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshape(1, -1)  # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T)  # (N, 4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1))  # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"])  # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(1, -1)  # (N, 2) -> (1, 2)

    def fit(self, X, Y, epochs=1, algo="GD", display_loss=False, eta=1, mini_batch_size=100, gamma=0.9, beta=0.9, beta1=0.9, beta2=0.9):

        if display_loss:
            loss = {}
        for num_epoch in tqdm(range(epochs), total=epochs, unit="epoch"):
            m = X.shape[0]

            if algo == "GD":
                self.grad(X, Y)
                for i in range(1, self.num_layers + 1):
                    self.params["W" + str(i)] -= eta * (self.gradients["dW" + str(i)] / m)
                    self.params["B" + str(i)] -= eta * (self.gradients["dB" + str(i)] / m)

            elif algo == "MGD":
                self.grad(X, Y)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = gamma * self.update_params["v_w" + str(i)] + eta * \
                        (self.gradients["dW" + str(i)] / m)
                    self.update_params["v_b" + str(i)] = gamma * self.update_params["v_b" + str(i)] + eta * \
                        (self.gradients["dB" + str(i)] / m)
                    self.params["W" + str(i)] -= self.update_params["v_w" + str(i)]
                    self.params["B" + str(i)] -= self.update_params["v_b" + str(i)]

            elif algo == "NAGD":
                temp_params = {}
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = gamma * self.prev_update_params["v_w" + str(i)]
                    self.update_params["v_b" + str(i)] = gamma * self.prev_update_params["v_b" + str(i)]
                    temp_params["W" + str(i)] = self.params["W" + str(i)] - self.update_params["v_w" + str(i)]
                    temp_params["B" + str(i)] = self.params["B" + str(i)] - self.update_params["v_b" + str(i)]
                self.grad(X, Y, temp_params)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = gamma * self.update_params["v_w" + str(i)] + eta * \
                        (self.gradients["dW" + str(i)] / m)
                    self.update_params["v_b" + str(i)] = gamma * self.update_params["v_b" + str(i)] + eta * \
                        (self.gradients["dB" + str(i)] / m)
                    self.params["W" + str(i)] -= eta * (self.update_params["v_w" + str(i)])
                    self.params["B" + str(i)] -= eta * (self.update_params["v_b" + str(i)])
                self.prev_update_params = self.update_params

            elif algo == "SGD":
                mini_batch_size = 1
                for k in range(0, m, mini_batch_size):
                    self.grad(X[k:k + mini_batch_size], Y[k:k + mini_batch_size])
                    for i in range(1, self.num_layers + 1):
                        self.params["W" + str(i)] -= eta * (self.gradients["dW" + str(i)] / mini_batch_size)
                        self.params["B" + str(i)] -= eta * (self.gradients["dB" + str(i)] / mini_batch_size)

            elif algo == "minibatch":
                for k in range(0, m, mini_batch_size):
                    self.grad(X[k:k + mini_batch_size], Y[k:k + mini_batch_size])
                    for i in range(1, self.num_layers + 1):
                        self.params["W" + str(i)] -= eta * (self.gradients["dW" + str(i)] / mini_batch_size)
                        self.params["B" + str(i)] -= eta * (self.gradients["dB" + str(i)] / mini_batch_size)

            if display_loss:
                Y_pred = self.predict(X)
                loss[num_epoch] = log_loss(np.argmax(Y, axis=1), Y_pred)

        if display_loss:
            plt.plot(loss.values(), '-o', markersize=5)
            plt.xlabel('Epochs')
            plt.ylabel('{} Log Loss'.format(algo))
            image_path = self.path + algo + '_Loss_Graph.png'
            print(image_path)
            plt.savefig(image_path)
            plt.show()

    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()

    def print_accuracy(self, X_train, Y_train, X_val, Y_val):
        Y_pred_train = self.predict(X_train)
        Y_pred_train = np.argmax(Y_pred_train, 1)
        Y_pred_val = self.predict(X_val)
        Y_pred_val = np.argmax(Y_pred_val, 1)
        accuracy_train = accuracy_score(Y_pred_train, Y_train)
        accuracy_val = accuracy_score(Y_pred_val, Y_val)
        print("Training accuracy: ", round(accuracy_train, 4))
        print("Validation accuracy: ", round(accuracy_val, 4))
        if False:
            plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_pred_train, cmap=dataset.my_cmap,
                        s=15 * (np.abs(np.sign(Y_pred_train - Y_train)) + .1))
            plt.show()

def main(args):
    W1 = np.random.randn(2, 2)
    W2 = np.random.randn(2, 4)
    ffn = FFNetwork(W1, W2)
    ffn.fit(dataset.X_train, dataset.y_OH_train, epochs=args.epochs, algo=args.algo_type,
            display_loss=args.display_loss, eta=args.learning_rate, mini_batch_size=args.mini_batch_size)
    ffn.print_accuracy(dataset.X_train, dataset.Y_train, dataset.X_val, dataset.Y_val)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, help='Enter the No of Epochs', required=True)
    parser.add_argument('--algo_type', type=str, help='Enter the Algorithm')
    parser.add_argument('--display_loss', type=bool, help='Set Display Loss to True to view Loss Graph')
    parser.add_argument('--learning_rate', type=float, help='Enter the Learning Rate')
    parser.add_argument('--mini_batch_size', type=int, help='Enter the Batch Size for Minibatch Algo')
    args = parser.parse_args()
    main(args)
