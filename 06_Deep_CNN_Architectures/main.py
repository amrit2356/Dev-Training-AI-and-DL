"""
Testing Models on CIDAR10 Dataset
"""
import argparse
import dataset
# from Models.ZFNet.ZFNet import ZFNet
from trainclassifier import ClassifierTrain


def main(args):
    training = ClassifierTrain(dataset.trainloader, dataset.testloader, learning_rate=args.learning_rate)
    training.train(args.epochs, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True, help='Enter the no of Epochs')
    parser.add_argument('--learning_rate', type=float, help='Enter the Learning Rate')
    parser.add_argument('--checkpoint_path', type=str, help='Enter the checkpoint path')
    args = parser.parse_args()
    main(args)
