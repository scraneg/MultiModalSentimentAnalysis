# Vision model for multi-modal sentiment analysis using ResNet-18, only get the features from the last layer

import torch.nn as nn
import torchvision.models as models


class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # delete the last layer, not identify the last layer
        modules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, input):
        # input shape: (batch_size, 3, 224, 224)
        output = self.resnet(input)
        output = output.reshape(output.size(0), -1)
        return output
    