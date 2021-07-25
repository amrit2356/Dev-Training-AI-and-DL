from input_parser import parse_input
from data_prep.utils import create_folder_structure
from data_prep.dataloader import Dataloader
from team_classifier_train import TeamClassifierTrain


def main(args):
    train_path, checkpoint_path, exports_path, run_path = create_folder_structure(args.output_path, args.project_name)
    data = Dataloader(args, args.batch_size, args.dataset_path, train_path, args.data_type)
    trainloader, testloader = data.data_loader()
    training = TeamClassifierTrain(trainloader, testloader, args.learning_rate, args.num_classes)
    training.train(args.epochs, checkpoint_path, exports_path, run_path)


if __name__ == "__main__":
    args = parse_input()
    main(args)