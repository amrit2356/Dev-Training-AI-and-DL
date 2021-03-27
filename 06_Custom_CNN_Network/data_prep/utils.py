"""Utils"""
import argparse
import os
from os.path import exists, join
from os import listdir, makedirs
import shutil


def create_path(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)


def path_validity_check(location_path, project_name):
    if not listdir(join(location_path, project_name)):
        return 0
    else:
        print('Files existing in Directory.')
        return 1


def create_folder_structure(location_path, project_name):
    """
    Creates the Output Folder Structure.

    params 'location_path':  leads to the project folder location.
    params 'project_name': gives the project name to the main directory.
    returns: the training dataset, checkpoint, and exported model path.try

    """
    if not exists(join(location_path, project_name)):
        makedirs(join(location_path, project_name))
        flag = 0

    else:
        flag = path_validity_check(location_path, project_name)

    if flag == 0:
        training_path = join(location_path, project_name, 'training')
        checkpoint_path = join(location_path, project_name, 'checkpoints')
        export_path = join(location_path, project_name, 'exports')
        makedirs(training_path)
        makedirs(checkpoint_path)
        makedirs(export_path)
        return training_path, checkpoint_path, export_path

    else:
        print('cannot Create Folder Structure, Files still exists...')
        return 1, 1, 1

def main(args):
    train_path, checkpoint_path, export_path = create_folder_structure(args.model_path, args.project_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    main(parser.parse_args())
