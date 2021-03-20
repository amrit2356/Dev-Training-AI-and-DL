from os.path import join
import torch
import torch.nn as nn
from tqdm import tqdm

class ClassifierTrain:
    def __init__(self, model, trainloader, testloader, learning_rate):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = trainloader
        self.test_loader = testloader
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-08)

    def train_epoch(self):
        loss = 0.0
        for data, labels in tqdm(self.train_loader):

            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            loss += loss.item() * data.size(0)
        return loss

    def val_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0.0
        total = 0
        with torch.no_grad():

            for data, labels in tqdm(self.test_loader):

                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_loss += loss.item() * data.size(0)

            val_loss += loss.item() * data.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = (correct / total) * 100

        return val_loss, acc

    def train(self, epochs, checkpoint_path):

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            validation_loss, validation_accuracy = self.val_epoch()

            train_loss = train_loss / len(self.train_loader.sampler)
            validation_loss = validation_loss / len(self.test_loader.sampler)

            print('Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f} Val Accuracy: {:.2f}%'.format(
                epoch, train_loss, validation_loss, validation_accuracy))

        print('Training complete!')
