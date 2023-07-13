import torch
import torch.nn as nn
from vision_model import VisionModel
from text_model import TextModel


# not freezed both models
class MultiModalModel(nn.Module):
    def __init__(self, n_classes):
        super(MultiModalModel, self).__init__()
        self.n_classes = n_classes
        self.text_model = TextModel()
        self.vision_model = VisionModel()
        self.linear = nn.Linear(768 + 512, self.n_classes)

    def forward(self, imgs, texts, atten_masks, token_type_ids):
        # imgs shape: (batch_size, 3, 224, 224)
        # texts shape: (batch_size, max_len)
        # atten_masks shape: (batch_size, max_len)
        # token_type_ids shape: (batch_size, max_len)
        text_output = self.text_model(texts, atten_masks, token_type_ids)
        vision_output = self.vision_model(imgs)
        output = torch.cat((text_output, vision_output), dim=1)
        output = self.linear(output)
        return output
