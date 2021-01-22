"""
Implementation of Perceptron using Class
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
from data_prep_class import DataPreparation


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    def predict(self, X):
        y = []
        for x in X:
            result = self.model(x)
            y.append(result)
        return np.array(y)

    def fit(self, X, Y, epochs=1, lr=1):
        self.w = np.ones(X.shape[1])
        self.b = 0
        accuracy = {}
        max_accuracy = 0

        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b + lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b - lr * 1
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if(accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb
        plt.plot(accuracy.values())
        plt.show()


def main():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    dataset = DataPreparation()
    perceptron = Perceptron()
    dataset.getdata(breast_cancer.data)
    dataset.gettarget(breast_cancer.target)
    data = dataset.dataset_generate(breast_cancer.feature_names)
    print(data.head())
    x_train, x_test, y_train, y_test = dataset.dataset_split(
        test_size=0.1, random_state=1)
    x_train = x_train.values
    x_test = x_test.values
    perceptron.fit(x_train, y_train, epochs=10000, lr=0.001)
    y_pred_train = perceptron.predict(x_train)
    print("Accuracy of Perceptron Model: {:.2f}".format(
        accuracy_score(y_pred_train, y_train)))


if __name__ == "__main__":
    main()
