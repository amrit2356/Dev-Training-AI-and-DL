import torch
import torchvision.transforms as transforms
from torchvision import datasets

def dataset():
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.2), shear=(-5, 5, -5, 5)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=128, shuffle=True, num_workers=2)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=128, shuffle=True, num_workers=2)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = dataset()
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(images.shape)
