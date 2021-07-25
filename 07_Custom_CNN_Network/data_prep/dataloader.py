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
from .detect_normalize_params import RunningAverage

class Dataloader:
    def __init__(self, args, batch_size, dataset_path, train_path, data_type):
        # Initialization  of DataLoader Attributes.
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.data_type = data_type
        self.normalize_params = RunningAverage()

        if self.data_type == 'custom':
            # Initialization of Dataset Preparation Class
            self.data_prep = DatasetPreparator(self.dataset_path, self.train_path)
            self.csv_path = self.data_prep.dataset_creator()
            running_mean, running_std_deviation = self.normalize_params.param_calculation(args, self.train_path, self.csv_path)
            # Getting Mean and Standard Deviation
            self.transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(40),
                transforms.Resize((160, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=running_mean, std=running_std_deviation),
            ])

            self.transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=running_std_deviation, std=running_std_deviation),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def data_loader(self):
        if self.data_type == 'custom':
            labels = pd.read_csv(self.csv_path)

            train_data, valid_data = train_test_split(labels, stratify=labels.cls, test_size=0.2)

            train = TeamDataset(train_data, self.train_path, self.transform_train)
            valid = TeamDataset(valid_data, self.train_path, self.transform_test)

            trainloader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, num_workers=4)
            testloader = DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False, num_workers=4)
            print('Trainloader & Testloader ready')

        elif self.data_type == 'CIFAR10':
            # Training Data from CIFAR10 Dataset
            train_data = datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=self.transform)
            trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2)

            # Testing Data from CIFAR10 Dataset
            test_data = datasets.CIFAR10(root=self.dataset_path, train=False, download=True, transform=self.transform)
            testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader

# Testing Code
"""
def main():
    data = Dataloader(512, './data', 'D:/Dev_Training_Folders/project/test-project/training', 'CIFAR10')
    trainloader, testloader = data.data_loader()
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)


if __name__ == "__main__":
    main()
"""