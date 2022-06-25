import config

import torch.nn as nn
import monai
import torch

def get_model():
    
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        ).to(config.DEVICE)
        
    return model
