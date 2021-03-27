"""
Implementation of AlexNet Model
"""
import torch
import torch.nn as nn

class Alexnet:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.running_loss = 0.0
        self.path = 'Exercises/05_Deep_CNN_Architectures/01_AlexNet/Loss_Graph/'

    def eval(self):
        return self.model.eval()

    def update_classifier(self, net_layer, neuron_input, neuron_output):
        self.model.classifier[net_layer] = nn.Linear(neuron_input, neuron_output)
