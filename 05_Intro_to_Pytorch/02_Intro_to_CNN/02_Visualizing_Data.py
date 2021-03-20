"""
Visualization of Image Data
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Loading a Training Dataset deom CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)


# Classes
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse','ship','truck')

# Getting Images with the dataset
dataiter = iter(train_loader)
images, labels = dataiter.next()

# print(images.shape)
# print(images[1].shape)
# print(labels.shape)


# DataType of single Image
img = images[1]
print(type(img))

# Conversion of image Tensor to numpy array
np_img = img.numpy()
print(np_img.shape)

np_img = np.transpose(np_img,(1, 2, 0))
print(np_img.shape)

# Visualization using Matplotlib
plt.figure(figsize = (1, 1))
plt.imshow(np_img)
plt.show()

def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))
print(' '.join(classes[labels[j]] for j in range(4)))
