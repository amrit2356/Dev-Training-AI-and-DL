"""
Testing of New Optimized FeedForward Neural Network
"""
import argparse
import numpy as np
import dataset
from opt_acti_init_ffnn import FFNNetwork


def main(args):
    X = dataset.x_train
    Y = dataset.y_OH_train
    model = FFNNetwork(init_method=args.init_method, activation_function=args.activation)
    model.fit(X, Y, epochs=args.epochs, algo=args.algo_type, display_loss=args.display_loss, eta=args.learning_rate, mini_batch_size=args.mini_batch_size)
    model.print_accuracy(dataset.x_train, dataset.x_val, dataset.y_train, dataset.y_val, scatter_plot=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_method', type=str, help='Set the Initialization Method', required=True)
    parser.add_argument('--activation', type=str, help='Set the Activation Function', required=True)
    parser.add_argument('--algo_type', type=str, help='Set the Learning Algorithm', required=True)
    parser.add_argument('--epochs', type=int, help='Set the no of Epochs')
    parser.add_argument('--learning_rate', type=float, help='Set the Learning Rate')
    parser.add_argument('--display_loss', type=bool, help='Set Display loss to True to view Loss Graph')
    parser.add_argument('--mini_batch_size', type=int, help='Set the batch size for minibatch Algo')
    args = parser.parse_args()
    main(args)
