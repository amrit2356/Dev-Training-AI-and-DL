"""
Dataloader.
"""
# import argparse
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .team_dataset import TeamDataset
from .prepare_classification_dataset import DatasetPreparator

class Dataloader:
    def __init__(self, batch_size, dataset_path, train_path):
        # Initialization  of DataLoader Attributes.
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_path = train_path

        # Initialization of Dataset Preparation Class
        self.data_prep = DatasetPreparator(self.dataset_path, self.train_path)
        self.csv_path = self.data_prep.dataset_creator()

        # __parameters_normalized() code
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

    def data_loader(self, data_type):

        if data_type == 'custom':
            labels = pd.read_csv(self.csv_path)

            train_data, valid_data = train_test_split(labels, stratify=labels.cls, test_size=0.2)

            train = TeamDataset(train_data, self.train_path, self.transform_train)
            valid = TeamDataset(valid_data, self.train_path, self.transform_test)

            trainloader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, num_workers=4)
            testloader = DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False, num_workers=4)
            print('Trainloader & Testloader ready')

        elif data_type == 'CIFAR10':
            # Training Data from CIFAR10 Dataset
            train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)

            # Testing Data from CIFAR10 Dataset
            test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader

# Testing Code
"""
def main(args):
    data = Dataloader(args.batch_size, args.filepath)
    trainloader, testloader = data.data_loader()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
"""
