"""
Dataloader.
"""

import pandas as pd
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .team_dataset import TeamDataset
from .prepare_classification_dataset import DatasetPreparator
from .detect_normalize_params import RunningAverage

class Dataloader:
    def __init__(self, config, batch_size, dataset_path, train_path):
        # Initialization  of DataLoader Attributes.
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_path = train_path
        self.normalize_params = RunningAverage()

        # Initialization of Dataset Preparation Class
        self.data_prep = DatasetPreparator(self.dataset_path, self.train_path)
        self.csv_path = self.data_prep.dataset_creator()

        # Getting Mean and Standard Deviation
        running_mean, running_std_deviation = self.normalize_params.param_calculation(config, self.train_path, self.csv_path)

        self.transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
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

    def data_loader(self):
        labels = pd.read_csv(self.csv_path)

        train_data, valid_data = train_test_split(labels, stratify=labels.cls, test_size=0.2)

        train = TeamDataset(train_data, self.train_path, self.transform_train)
        valid = TeamDataset(valid_data, self.train_path, self.transform_test)

        trainloader = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(dataset=valid, batch_size=self.batch_size, shuffle=False, num_workers=4)
        print('Trainloader & Testloader ready')
        return trainloader, testloader
