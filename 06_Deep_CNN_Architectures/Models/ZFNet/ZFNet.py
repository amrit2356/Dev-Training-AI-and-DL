"""
Implementation of ZFNet using Pytorch
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ZFNet(nn.Module):
    def __init__(self):
        super(ZFNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=(4, 4)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(1024, 10)
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = ''

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x
