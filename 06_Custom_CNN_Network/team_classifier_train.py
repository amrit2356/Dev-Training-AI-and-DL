from os.path import join, isfile, exists
from os import makedirs
import torch
import torch.nn as nn
from tqdm import tqdm
from models.team_classifier_cnn import TeamClassifier as Model

class TeamClassifierTrain:
    def __init__(self, trainloader, testloader, learning_rate):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Model().to(self.device)
        self.train_loader = trainloader
        self.test_loader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-08)
        print("Model, Loss function & Optimizer Initialized...")
        print("Training Starting....")

    def __train_epoch(self):
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

                loss = self.criterion(outputs, labels)
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

        filename = join(checkpoint_path, 'team_classifier_epoch_{}_val_acc{}.pth'.format(epoch, val_acc))
        torch.save({
            'model': self.model,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_loss': train_loss}, filename)
        return filename

    def __export_model(self, checkpoint_path, export_path):
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        filename = join(export_path, 'team_classifier_model.pt')
        torch.save(model.state_dict(), filename)
        print('Model Exported in this path: {}'.format(export_path))

    def load_model(self, model_path):
        if isfile(model_path):
            weights = torch.load(model_path)
            self.model.load_state_dict(weights['state_dict'])
            self.model.to(self.device)
            self.model.eval()

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
                model_path = self.__save_model(checkpoint_path, epoch, train_loss, val_acc_best)
            print('Checkpoint saved in this path: {}'.format(model_path))

        self.__export_model(model_path, export_path)
        print('Training complete!')

    def mlflow_logging(self):
        pass
