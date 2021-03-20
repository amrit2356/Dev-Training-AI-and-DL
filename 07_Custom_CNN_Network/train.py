import argparse
from team_classifier_train import TeamClassifierTrain
from data_prep.dataloader import Dataloader

def main(args):
    data = Dataloader(args.batch_size, args.dataset_path)
    trainloader, testloader = data.data_loader()
    training = TeamClassifierTrain(trainloader, testloader, args.learning_rate)
    training.train(args.epochs, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='/home/edisn/Pytorch_CNN_Training/Dev-Training-DL/Exercises/06_Custom_CNN_Network/checkpoints')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Enter the Learning Rate")
    args = parser.parse_args()
    main(args)
