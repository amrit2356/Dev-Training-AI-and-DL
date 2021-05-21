"""
Implementation of AlexNet Model
"""
import torch
import torch.nn as nn

class VggNet:
    def __init__(self):
        torch.hub.set_dir('D:/Dev_Training_Folders/pytorch_models')
        # Use any of these variants
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg11_bn', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg13_bn', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16_bn', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19_bn', pretrained=True)
        self.path = 'Exercises/05_Deep_CNN_Architectures/VggNet/Loss_Graph/'

    def update_classifier(self, net_layer, neuron_input, neuron_output):
        self.model.classifier[net_layer] = nn.Linear(neuron_input, neuron_output)
