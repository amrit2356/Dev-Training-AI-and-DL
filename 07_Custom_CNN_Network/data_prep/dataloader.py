"""
Generic Dataloader.
"""
import argparse
from os.path import join, exists

import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .team_dataset import TeamDataset
from .prepare_classification_dataset import DatasetPreparator

class Dataloader:
    def __init__(self, batch_size, dataset_path):
        self.data_csv = None
        self.image_path = None

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_path = '/home/edisn/Pytorch_CNN_Training/Dev-Training-DL/Exercises/06_Custom_CNN_Network/training_data/'
        self.data_prep = DatasetPreparator()

        # Getting Mean and Standard Deviation
        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(40),
            transforms.Resize((160, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.438, 0.479, 0.336], std=[0.150, 0.155, 0.152]),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.438, 0.479, 0.335], std=[0.150, 0.155, 0.152]),
        ])

    def __dataset_prep(self):
        return self.data_prep.dataset_creator(self.dataset_path, self.train_path)

    def __parameters_normalize():
        # To check whether parameters are normalized
        pass

    def data_loader(self):
        self.data_csv = self.__dataset_prep()
        labels = pd.read_csv(self.data_csv)

        train_data, valid_data = train_test_split(labels, stratify=labels.cls, test_size=0.2)

        train = TeamDataset(train_data, self.train_path, self.transform_train)
        valid = TeamDataset(valid_data, self.train_path, self.transform_test)

        trainloader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False, num_workers=4)
        print('Trainloader & Testloader ready')
        return trainloader, testloader


def main(args):
    data = Dataloader(args.batch_size, args.filepath)
    trainloader, testloader = data.data_loader()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='/home/edisn/Pytorch_CNN_Training/Dev-Training-DL/Exercises/06_Custom_CNN_Network/dataset')
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    main(args)
