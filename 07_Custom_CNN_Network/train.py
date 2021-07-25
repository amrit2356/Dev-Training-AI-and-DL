import argparse
from data_prep.dataloader import Dataloader
from data_prep.utils import create_folder_structure
from input_parser import parse_input
from team_classifier_train import TeamClassifierTrain


def main(config):
    # Project Attributes
    project_path = config.project_details.project_path
    dataset_path = config.project_details.dataset_path
    project_name = config.project_details.project_name

    # Training Attributes
    batch_size = config.training_parameters.batch_size
    learning_rate = config.training_parameters.learning_rate
    num_classes = config.training_parameters.num_classes
    num_epochs = config.training_parameters.num_epochs

    # Torch Device
    device = config.normalization_param.device
    
    # Setting the Training, Checkpoint, Export, Runs Folder Path
    train_path, checkpoint_path, exports_path, run_path = create_folder_structure(project_path, project_name)
    
    # Preparing the Dataset for Training
    data = Dataloader(config, batch_size, dataset_path, train_path)
    trainloader, testloader = data.data_loader()
    
    # Initialize Training Class
    training = TeamClassifierTrain(device, trainloader, testloader, learning_rate, num_classes)
    training.train(num_epochs, checkpoint_path, exports_path, run_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default='07_Custom_CNN_Network/config.json')
    config = parse_input(parser.parse_args())
    main(config)
