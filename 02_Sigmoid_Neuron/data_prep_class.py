import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreparation:
    def __init__(self):
        self.w = None
        self.b = None
        self.threshold = None
        self.data = None

    def readcsv(self, csvpath):
        data = pd.read_csv(csvpath)
        self.data = data
        return self.data

    def generate_binary_threshold(self, data, threshold=5.0):
        x = data.drop(data.columns[-1], axis=1)
        y = data[data.columns[-1]]
        data['Class'] = (self.data[self.data.columns[-1]] >= threshold).astype(np.int)
        print(data['Class'].value_counts(normalize=True))
        Y_binarised = data['Class'].values
        return x, y, threshold, Y_binarised

    def dataset_split(self, x, y, stratify, random_state=1):
        if self.data is None:
            return print("Data is not available to split")
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, stratify=stratify,
                random_state=random_state)
        return x_train, x_test, y_train, y_test

    def standardization(self, X_train, X_test, Y_train, Y_test, threshold):
        # Using the Standard Scaler
        scaler = StandardScaler()
        X_scaled_train = scaler.fit_transform(X_train)
        X_scaled_test = scaler.transform(X_test)

        # Using the MinMax Scaler
        minmax_scaler = MinMaxScaler()
        Y_scaled_train = minmax_scaler.fit_transform(Y_train.values.reshape(-1, 1))
        Y_scaled_test = minmax_scaler.transform(Y_test.values.reshape(-1, 1))
        scaled_threshold = list(minmax_scaler.transform(np.array([threshold]).reshape(1, -1)))[0][0]

        return X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test, scaled_threshold

    def threshold_scaling(self, Y_scaled_train, Y_scaled_test, scaled_threshold):
        Y_binarised_train = (Y_scaled_train > scaled_threshold).astype("int").ravel()
        Y_binarised_test = (Y_scaled_test > scaled_threshold).astype("int").ravel()
        return Y_binarised_train, Y_binarised_test


def main():
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

    # Generation of Binarized Values for Y Scaled Train and Test data.
    Y_binarised_train, Y_binarised_test = dataset_obj.threshold_scaling(y_sca_tn, y_sca_tes, sca_thresh)


if __name__ == "__main__":
    main()
