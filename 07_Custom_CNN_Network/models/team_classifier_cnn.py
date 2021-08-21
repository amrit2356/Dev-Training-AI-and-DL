import torch
import torch.nn as nn
import torch.nn.functional as F

class TeamClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TeamClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.dropout = nn.Dropout2d(p=0.50)

        self.fc_layer1 = nn.Linear(1024, 512)
        self.fc_layer2 = nn.Linear(512, 256)
        self.fc_layer3 = nn.Linear(256, num_classes)

        self.training = True

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv3(x)), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv4(x)), 2))
        x = x.view(x.shape[0], -1)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x


if __name__ == "__main__":
    x = torch.rand([4, 3, 160, 64])
    net = TeamClassifier()
    out = net(x)
    print("Output Dimension {}:".format(out.shape))
