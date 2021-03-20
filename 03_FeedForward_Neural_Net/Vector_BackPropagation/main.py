"""
Module Comparison between Scalar and Vectorized Backpropagation.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from scalar_backprop import FF_MultiClass_Scalar

import dataset
from vectorised_weights_backprop import FF_MultiClass_WeightVectorised
from vectorized_weights_inputs_backprop import \
    FF_MultiClass_InputWeightVectorised

from sklearn.metrics import accuracy_score


def main():
    W1 = np.random.randn(2, 2)
    W2 = np.random.randn(2, 4)
    models_init = [FF_MultiClass_Scalar(W1, W2), FF_MultiClass_WeightVectorised(
        W1, W2), FF_MultiClass_InputWeightVectorised(W1, W2)]
    models = []
    for idx, model in enumerate(models_init, start=1):
        tic = time.time()
        ffsn_multi_specific = model
        ffsn_multi_specific.fit(dataset.X_train, dataset.y_OH_train,
                                epochs=2000, learning_rate=0.55, display_loss=True)
        models.append(ffsn_multi_specific)
        toc = time.time()
        print("Time taken by model {}: {:.3f}".format(idx, toc - tic))

    for idx, model in enumerate(models, start=1):
        Y_pred_train = model.predict(dataset.X_train)
        Y_pred_train = np.argmax(Y_pred_train, 1)

        Y_pred_val = model.predict(dataset.X_val)
        Y_pred_val = np.argmax(Y_pred_val, 1)

        accuracy_train = accuracy_score(Y_pred_train, dataset.Y_train)
        accuracy_val = accuracy_score(Y_pred_val, dataset.Y_val)

        print("Model {}".format(idx))
        print("Training accuracy", round(accuracy_train, 2))
        print("Validation accuracy", round(accuracy_val, 2))

    plt.scatter(dataset.X_train[:, 0], dataset.X_train[:, 1], c=Y_pred_train,
                cmap=dataset.my_cmap, s=15 * (np.abs(np.sign(Y_pred_train - dataset.Y_train)) + .1))
    plt.show()


if __name__ == "__main__":
    main()
