from os import makedirs
from os.path import exists, isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.spatial_transformer_network import SpatialTransformerNet as Model


class STNTrain:
    def __init__(self, trainloader, testloader, learning_rate):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)
        self.train_loader = trainloader
        self.test_loader = testloader
        print('Data Ready for Training...')
        # self.criterion = F.nll_loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        print("Model, Loss function & Optimizer Initialized...")
        print("Training Starting....")

    def __train_epoch(self):
        loss = 0.0
        for data, labels in tqdm(self.train_loader):

            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = F.nll_loss(outputs, labels)

            loss.backward()
            self.optimizer.step()
            loss += loss.item() * data.size(0)
        return loss

    def __val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0
        with torch.no_grad():

            for data, labels in tqdm(self.test_loader):

                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)

                loss = F.nll_loss(outputs, labels, reduction='sum')
                _, predicted = torch.max(outputs.data, 1)
                val_loss += loss.item() * data.size(0)

            val_loss += loss.item() * data.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = (correct / total) * 100

        return val_loss, acc

    def __save_model(self, checkpoint_path, epoch, train_loss, val_acc):
        if not exists(checkpoint_path):
            makedirs(checkpoint_path)

        filename = join(checkpoint_path, 'spatial_transformer_networks_best.pth')
        torch.save({
            'model': self.model,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_loss': train_loss}, filename)

    def load_model(self, model_path):
        if isfile(model_path):
            weights = torch.load(model_path)
            self.model.load_state_dict(weights['state_dict'])
            self.model.to(self.device)
            self.model.eval()

    def __export_model(self, checkpoint_path, export_path):
        # export_path = '/home/edisn/Pytorch_CNN_Training/Dev-Training-DL/Exercises/06_Custom_CNN_Network/exports/'
        checkpoint = torch.load(join(checkpoint_path, 'spatial_transformer_networks_best.pth'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        filename = join(export_path, 'spatial_transformer_network.pt')
        torch.save(model.state_dict(), filename)
        print('Model Exported in this path: {}'.format(export_path))

    def train(self, epochs, checkpoint_path, export_path):
        train_loss_best = 0.0
        val_loss_best = 0.0
        val_acc_best = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = self.__train_epoch()
            validation_loss, validation_accuracy = self.__val_epoch()

            train_loss = train_loss / len(self.train_loader.sampler)
            validation_loss = validation_loss / len(self.test_loader.sampler)

            print('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f} Val Accuracy: {:.2f}%'.format(
                epoch, train_loss, validation_loss, validation_accuracy))

            if val_loss_best < validation_loss or val_acc_best < validation_accuracy:
                val_loss_best = round(validation_loss, 3)
                val_acc_best = round(validation_accuracy, 3)
                self.__save_model(checkpoint_path, epoch, train_loss, val_acc_best)

        self.__export_model(checkpoint_path, export_path)
        print('Training complete!')

    def mlflow_logging(self):
        pass
