"""
Data Loading in Pytorch
"""
import torch
import torchvision
import torchvision.transforms as transforms

# Loading a Training Dataset deom CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


# Classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Getting Images with the dataset
dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)
print(images[1].shape)
print(labels.shape)
