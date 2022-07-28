import torch
import torch.nn as nn

import config

from transformers import SegformerForSemanticSegmentation

class MixUpSample(nn.Module):
    def __init__( self, scale_factor=4):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing *F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1-self.mixing )*F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x

class HubmapModel(nn.Module):
    def __init__(self):
        super(HubmapModel, self).__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(config.MODEL_PATH,
                                                         num_labels=1,ignore_mismatched_sizes=True)
        self.mixup = MixUpSample()
    def forward(self, image):
        img_segs = self.model(image)

        upsampled_logits = self.mixup(img_segs.logits)
        return upsampled_logits
