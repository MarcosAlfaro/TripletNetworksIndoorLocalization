"""
THIS SCRIPT CONTAINS A CLASS THAT DEFINES THE NETWORK ARCHITECTURE (VGG16)
The training scripts will call this function
"""


import torch.nn as nn
import torch
from torchvision.models import VGG16_Weights

vgg16 = torch.hub.load('pytorch/vision:v0.12.0', 'vgg16', weights=VGG16_Weights.DEFAULT)


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.cnn1 = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * 512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def forward(self, x):
        out = self.cnn1(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out
