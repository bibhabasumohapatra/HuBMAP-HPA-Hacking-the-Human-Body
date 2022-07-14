import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def HubmapModel():

    model = smp.Unet(encoder_name="efficientnet-b5", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1).to("cuda")
    
    return model
