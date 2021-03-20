"""
Implementation of Single CNN Layer
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class FirstCNN_v2(nn.Module):
    def __init__(self):
        super(FirstCNN_v2, self).__init__()
        self.model = nn.Sequential( 
            nn.Conv2d(3, 6, 5),          # (N, 3, 32, 32) -> (N, 6, 28, 28)
            nn.AvgPool2d(2, stride=2),   # (N, 6, 28, 28) -> (N, 6, 14, 14)
            nn.Conv2d(6, 16, 5),         # (N, 6, 14, 14) -> (N, 16, 10, 10)
            nn.AvgPool2d(2, stride=2)   # (N, 16, 10, 10) -> (N, 16, 5, 5))
        )

    def forward(self, x):
        x = self.model(x)
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

    net = FirstCNN_v2()
    out = net(images)
    print(out.shape)

    for param in net.parameters():
        print(param.shape)

    plt.imshow(out[0, 0, :, :].detach().numpy())   

if __name__ == "__main__":
    main()