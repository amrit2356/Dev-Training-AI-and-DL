"""
Implementation of the FFNN with Multiple Activation Functions &
Initializations.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

import dataset


class FFNNetwork:
    def __init__(self, init_method='random', activation_function='sigmoid'):
        # Input Layer's weights and bias values
        self.params = {}
        self.params_h = []
        # No of Hidden Layers
        self.num_layers = 2
        # Size of each Layer
        self.layer_sizes = [2, 2, 4]
        # Selection of Initialization Method
        self.init_method = init_method
        # Selection of activation Layer
        self.activation_function = activation_function
        # Leaky Slope for Leaky-Relu Activation Function
        self.leaky_slope = 0.1
        # Folder Path for Loss Graphs
        self.path = 'Exercises/03_Optimization_Algorithms/02_Better_Activation_Initialization_Algos/Loss_Graphs'

        np.random.seed(0)
        # Initialization Methods
        # Zero Value Init Method
        if init_method == "zeros":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] = np.zeros(
                    (self.layer_sizes[i - 1], self.layer_sizes[i]))
                self.params["B" + str(i)] = np.zeros((1, self.layer_sizes[i]))
        # Random Value Init Method
        elif init_method == "random":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] = np.random.randn(
                    self.layer_sizes[i - 1], self.layer_sizes[i])
                self.params["B" + str(i)] = np.random.randn(1,
                                                            self.layer_sizes[i])
        # He Init Method
        elif init_method == "he":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) \
                    / np.sqrt(2 / self.layer_sizes[i - 1])
                self.params["B" + str(i)] = np.random.randn(1,
                                                            self.layer_sizes[i])
        # Xavier Init Method
        elif init_method == "xavier":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] = np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) \
                    / np.sqrt(1 / self.layer_sizes[i - 1])
                self.params["B" + str(i)] = np.random.randn(1,
                                                            self.layer_sizes[i])

        self.gradients = {}
        self.update_params = {}
        self.prev_update_params = {}

        for i in range(1, self.num_layers + 1):
            self.update_params["v_w" + str(i)] = 0
            self.update_params["v_b" + str(i)] = 0
            self.update_params["m_w" + str(i)] = 0
            self.update_params["m_b" + str(i)] = 0
            self.prev_update_params["v_w" + str(i)] = 0
            self.prev_update_params["v_b" + str(i)] = 0

    def forward_activation(self, X):
        """
        Multiple Activations Functions:(logistic, tanh, relu, leaky-relu)
        """
        if self.activation_function == "logistic":
            return 1.0 / (1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0, X)
        elif self.activation_function == "leaky-relu":
            return np.maximum(self.leaky_slope * X, X)

    def softmax(self, y):
        exps = np.exp(y)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def forward_pass(self, X, params=None):
        if params is None:
            params = self.params

        # 1st Hidden layer
        self.A1 = np.matmul(X, params["W1"]) + params["B1"]
        self.H1 = self.forward_activation(self.A1)

        # Output Layer
        self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"]
        self.H2 = self.softmax(self.A2)

        return self.H2

    def grad_activation(self, X):
        """
        Gradients for each Activation Functions:(logistic, tanh, relu, leaky-relu)
        """
        if self.activation_function == "logistic":
            return X * (1 - X)
        elif self.activation_function == "tanh":
            return (1 - np.square(X))
        elif self.activation_function == "relu":
            return 1.0 * (X > 0)
        elif self.activation_function == "leaky-relu":
            d = np.zeros_like(X)
            d[X <= 0] = self.leaky_slope
            d[X > 0] = 1
            return d

    def grad(self, X, Y, params=None):
        if params is None:
            params = self.params

        self.forward_pass(X, params=params)
        m = X.shape[0]
        self.gradients["dA2"] = self.H2 - Y  # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"])  # (2, N) * (N, 4) -> (2, 4)
        self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshape(1, -1)  # (N, 4) -> (1, 4)
        self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T)  # (N, 4) * (4, 2) -> (N, 2)
        self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1))  # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"])  # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(1, -1)  # (N, 2) -> (1, 2)

    def fit(self, X, Y, epochs=1, algo="GD", display_loss=False, eta=1, mini_batch_size=100, eps=1e-8, beta=0.9, beta1=0.9, beta2=0.9,
            gamma=0.9):

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

            elif algo == "Adagrad":
                self.grad(X, Y)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] += (self.gradients["dW" + str(i)] / m)**2
                    self.update_params["v_b" + str(i)] += (self.gradients["dB" + str(i)] / m)**2

                    self.params["W" + str(i)] -= (eta / (np.sqrt(self.update_params["v_w" + str(i)]) + eps)) * \
                        (self.gradients["dW" + str(i)] / m)
                    self.params["B" + str(i)] -= (eta / (np.sqrt(self.update_params["v_b" + str(i)]) + eps)) * \
                        (self.gradients["dB" + str(i)] / m)

            elif algo == "RMSProp":
                self.grad(X, Y)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = beta * self.update_params["v_w" + str(i)] + (1 - beta) * \
                        ((self.gradients["dW" + str(i)] / m)**2)
                    self.update_params["v_b" + str(i)] = beta * self.update_params["v_b" + str(i)] + (1 - beta) * \
                        ((self.gradients["dB" + str(i)] / m)**2)
                    self.params["W" + str(i)] -= (eta / (np.sqrt(self.update_params["v_w" + str(i)] + eps))) * \
                        (self.gradients["dW" + str(i)] / m)
                    self.params["B" + str(i)] -= (eta / (np.sqrt(self.update_params["v_b" + str(i)] + eps))) * \
                        (self.gradients["dB" + str(i)] / m)

            elif algo == "Adam":
                self.grad(X, Y)
                num_updates = 0
                for i in range(1, self.num_layers + 1):
                    num_updates += 1
                    self.update_params["m_w" + str(i)] = beta1 * self.update_params["m_w" + str(i)] + (1 - beta1) * \
                        (self.gradients["dW" + str(i)] / m)
                    self.update_params["v_w" + str(i)] = beta2 * self.update_params["v_w" + str(i)] + (1 - beta2) * \
                        ((self.gradients["dW" + str(i)] / m) ** 2)

                    m_w_hat = self.update_params["m_w" + str(i)] / (1 - np.power(beta1, num_updates))
                    v_w_hat = self.update_params["v_w" + str(i)] / (1 - np.power(beta2, num_updates))
                    self.params["W" + str(i)] -= (eta / np.sqrt(v_w_hat + eps)) * m_w_hat

                    self.update_params["m_b" + str(i)] = beta1 * self.update_params["m_b" + str(i)] + (1 - beta1) * \
                        (self.gradients["dB" + str(i)] / m)
                    self.update_params["v_b" + str(i)] = beta2 * self.update_params["v_b" + str(i)] + (1 - beta2) * \
                        ((self.gradients["dB" + str(i)] / m) ** 2)

                    m_b_hat = self.update_params["m_b" + str(i)] / (1 - np.power(beta1, num_updates))
                    v_b_hat = self.update_params["v_b" + str(i)] / (1 - np.power(beta2, num_updates))

                    self.params["B" + str(i)] -= (eta / np.sqrt(v_b_hat + eps)) * m_b_hat

            if display_loss:
                Y_pred = self.predict(X)
                loss[num_epoch + 1] = log_loss(np.argmax(Y, axis=1), Y_pred)
                self.params_h.append(np.concatenate((self.params['W1'].ravel(), self.params['W2'].ravel(), self.params['B1'].ravel(), self.params['B2'].ravel())))

        if display_loss:
            """
            Folder Structure ->Loss_Graph/init_method/activations/loss_graph.png
            """
            folder_path = os.path.join(self.path, self.init_method, self.activation_function)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.plot(loss.values(), '-o', markersize=5)
            title = '{} Initializer, {} Activation, {} Learning Algo. Loss Graph'.format(self.init_method, self.activation_function, algo)
            plt.title(title)
            plt.xlabel('Epochs')
            plt.ylabel('Log Loss')
            image_path = folder_path + '/{}_{}_{}'.format(self.init_method, self.activation_function, algo) + '_Loss_Graph.png'
            plt.savefig(image_path)
            plt.show()

    def predict(self, X):
        Y_pred = self.forward_pass(X)
        return np.array(Y_pred).squeeze()

    def print_accuracy(self, X_train, X_val, Y_train, Y_val, scatter_plot=False, plot_scale=0.1):
        Y_pred_train = self.predict(X_train)
        Y_pred_train = np.argmax(Y_pred_train, 1)
        Y_pred_val = self.predict(X_val)
        Y_pred_val = np.argmax(Y_pred_val, 1)
        accuracy_train = accuracy_score(Y_pred_train, Y_train)
        accuracy_val = accuracy_score(Y_pred_val, Y_val)
        print("Training accuracy", round(accuracy_train, 4))
        print("Validation accuracy", round(accuracy_val, 4))

        if scatter_plot:
            plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_pred_train, cmap=dataset.my_cmap, s=15 * (np.abs(np.sign(Y_pred_train - Y_train)) + .1))
            plt.show()
