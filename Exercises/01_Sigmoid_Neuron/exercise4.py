"""
Image Classification using Sigmoid Neuron: Level 2
"""
import os
import argparse
from PIL import Image

import numpy as np
import pandas as pd
from exercise2 import Sigmoid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def read_all(folder_path, key_prefix=""):
    '''
    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
    '''
    # print("Reading:")
    images = {}
    files = os.listdir(folder_path)
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        image_index = key_prefix + file_name[:-4]
        image = Image.open(file_path)
        image = image.convert("L")
        images[image_index] = np.array(image.copy()).flatten()
        image.close()
    return images


def main(args):
    LEVEL = 'level_2'
    languages = ['ta', 'hi', 'en']
    trainPath = args.dataset_path + LEVEL + "_train/" + LEVEL + "/"
    testPath = args.dataset_path + LEVEL + "_test/kaggle_" + LEVEL
    images_train = read_all(trainPath + "background", key_prefix='bkg_')
    for language in languages:
        images_train.update(
            read_all(trainPath + language, key_prefix=language + "_"))
    images_test = read_all(testPath, key_prefix='')
    # print(len(images_train))
    # print(len(images_test))

    X_train = []
    Y_train = []
    for key, values in images_train.items():
        X_train.append(values)
        if key[:4] == 'bkg_':
            Y_train.append(0)
        else:
            Y_train.append(1)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    id_test = []
    X_test = []

    for key, values in images_test.items():
        id_test.append(int(key))
        X_test.append(values)
    X_test = np.array(X_test)

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    sn = Sigmoid()
    print("Training Model...")
    sn.fit(X_scaled_train, Y_train, learning_rate=args.learning_rate,
           epochs=args.epochs, loss_type=args.loss_type, display_loss=args.display_loss)

    Y_pred_train = sn.predict(X_scaled_train)
    Y_pred_test = sn.predict(X_scaled_test)
    threshold = 0.5
    Y_pred_binarised_train = (Y_pred_train >= threshold).astype("int").ravel()
    Y_pred_binarised_test = (Y_pred_test >= threshold).astype("int").ravel()
    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
    print("Train Accuracy : {:.3f} ".format(accuracy_train))

    Y_pred_test = sn.predict(X_scaled_test)
    Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()

    submission = {}
    submission['ImageId'] = id_test
    submission['Class'] = Y_pred_binarised_test

    submission = pd.DataFrame(submission)
    submission = submission[['ImageId', 'Class']]
    submission = submission.sort_values(['ImageId'])
    submission.to_csv("submission_{}.csv".format(LEVEL), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Enter Dataset File Path', required=True)
    parser.add_argument('--learning_rate', type=float,
                        help='Enter the Learning Rate[0, 1]')
    parser.add_argument('--epochs', type=int,
                        help='Enter the Number of Epochs')
    parser.add_argument('--loss_type', type=str,
                        help='Choose Loss Type: MSE or CE')
    parser.add_argument('--display_loss', type=bool,
                        help='Choose Loss Type: MSE or CE')
    args = parser.parse_args()
    main(args)
