import argparse
from data_prep.dataloader import Dataloader
from data_prep.utils import create_folder_structure
from input_parser import parse_input
from team_classifier_train import TeamClassifierTrain


def main(config):
    train_path, checkpoint_path, exports_path, run_path = create_folder_structure(config.project_details.project_path, config.project_details.project_name)
    data = Dataloader(config, config.training_parameters.batch_size, config.project_details.dataset_path, train_path, 'custom')
    trainloader, testloader = data.data_loader()
    training = TeamClassifierTrain(trainloader, testloader, config.training_parameters.learning_rate, config.training_parameters.num_classes)
    training.train(config.training_parameters.num_epochs, checkpoint_path, exports_path, run_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='07_Custom_CNN_Network/config.json')
    config = parse_input(parser.parse_args())
    main(config)
