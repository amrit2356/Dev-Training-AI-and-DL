"""
Training & Testing of LeNet Neural Network Model.
"""
from LeNet import LeNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time as time


def main():
    # Setting the processing to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Loading Training and Testing Dataset from CIFAR10
    batch_size = 128
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initializing the LeNet Object
    net = LeNet().to(device)
    # Initializing the Cross Entropy Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Initializing the Adam Optimizer
    opt = optim.Adam(net.parameters())

    start_time = time.time()
    # Training the Model
    net.fit(train_loader, test_loader, max_epochs=16, opt=opt, loss_fn=loss_fn)
    duration = time.time() - start_time
    print("Duration of Model: {:.2f} secs".format(duration))

if __name__ == "__main__":
    main()
