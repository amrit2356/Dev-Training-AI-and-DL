import pandas as pd
from sklearn.model_selection import train_test_split
# import sklearn.datasets


class DataPreparation:
    def __init__(self):
        self.x = None
        self.y = None
        self.data = None

    def getdata(self, x):
        self.x = x

    def gettarget(self, y):
        self.y = y

    def dataset_generate(self, column_header):
        self.data = pd.DataFrame(self.x, columns=column_header)
        self.data['class'] = self.y
        return self.data

    def dataset_split(self, test_size=0.2, random_state=0):
        if self.data is None:
            return print("Data is not available to split")
        else:
            x = self.data.drop(self.data.columns[-1], axis=1)
            y = self.data[self.data.columns[-1]]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, stratify=y,
                random_state=random_state)
        return x_train, x_test, y_train, y_test

    def dataset_binarization(self, x_train, x_test):
        x_binarized_train = x_train.apply(pd.cut, bins=2, labels=[1, 0])
        x_binarized_test = x_test.apply(pd.cut, bins=2, labels=[1, 0])
        return x_binarized_train.values, x_binarized_test.values

# def main():
#     breast_cancer = sklearn.datasets.load_breast_cancer()
#     test_data = DataPreparation()
#     test_data.getdata(breast_cancer.data)
#     print(test_data.x)
#     test_data.gettarget(breast_cancer.target)
#     print(test_data.y)
#     data = test_data.dataset_generate(breast_cancer.feature_names)
#     print(data.head())
#     x_train, x_test, y_train, y_test = test_data.dataset_split(
#         test_size=0.1, random_state=1)
#     print(test_data.x.shape, x_train.shape, x_test.shape)
#     print(test_data.y.shape, y_train.shape, y_test.shape)

#     x_binarized_train, x_binarized_test = test_data.dataset_binarization(
#         x_train, x_test)


# if __name__ == "__main__":
#     main()
