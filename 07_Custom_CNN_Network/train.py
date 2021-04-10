import argparse
from .team_classifier_train import TeamClassifierTrain
from .data_prep.utils import create_folder_structure
from .data_prep.dataloader import Dataloader

def main(args):
    train_path, checkpoint_path, exports_path = create_folder_structure(args.output_path, args.project_name)
    data = Dataloader(args.batch_size, args.dataset_path, train_path)
    trainloader, testloader = data.data_loader()
    training = TeamClassifierTrain(trainloader, testloader, args.learning_rate)
    training.train(args.epochs, checkpoint_path, exports_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Enter the Learning Rate")
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    main(args)
