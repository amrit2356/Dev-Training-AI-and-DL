"""
Implementation of Single CNN Layer
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class FirstCNN(nn.Module):
    def __init__(self):
        super(FirstCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=(1,1), stride=(2,2))

    def forward(self, x):
        x = self.conv1(x)
        return x

def main():
    # Loading a Training Dataset deom CIFAR10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


    # Classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse','ship','truck')

    # Getting Images with the dataset
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    net = FirstCNN()
    out = net(images)
    print(out.shape)

    for param in net.parameters():
        print(param.shape)

    plt.imshow(out[0, 0, :, :].detach().numpy())   

if __name__ == "__main__":
    main()