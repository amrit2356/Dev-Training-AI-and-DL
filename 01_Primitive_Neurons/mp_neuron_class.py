"""
Implementation of MP Neuron using Classes and Objects.
"""
import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
from data_prep_class import DataPreparation


class MPNeuron:
    """
    This is a Implementation of MP Neuron using classes.
    It has the following functions:
        - model() -> to build the MP Neuron
        - predict() -> use the training dataset to train the model
        - fit() -> to use the training and testing data to find the
                   best accuracy score
    """

    def __init__(self):
        self.b = None

    def model(self, x):
        return (sum(x) >= self.b)

    def predict(self, X):
        y = []
        for x in X:
            result = self.model(x)
            y.append(result)
        return np.array(y)

    def fit(self, X, Y):
        accuracy = {}

        for b in range(X.shape[1]+1):
            self.b = b
            y_pred = self.predict(X)
            accuracy[b] = accuracy_score(y_pred, Y)
        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b

        print('Optimal Value of b is: {}'.format(best_b))
        print('Highest Accuracy is: {:.2f}'.format(accuracy[best_b]))


def main():
    breast_cancer = sklearn.datasets.load_breast_cancer()
    dataset_prep = DataPreparation()
    dataset_prep.getdata(breast_cancer.data)
    dataset_prep.gettarget(breast_cancer.target)
    data = dataset_prep.dataset_generate(breast_cancer.feature_names)
    print(data.head())
    x_train, x_test, y_train, y_test = dataset_prep.dataset_split(
        test_size=0.1, random_state=1)
    x_binarized_train, x_binarized_test = dataset_prep.dataset_binarization(
        x_train, x_test)
    mp = MPNeuron()
    mp.fit(x_binarized_train, y_train)


if __name__ == "__main__":
    main()
