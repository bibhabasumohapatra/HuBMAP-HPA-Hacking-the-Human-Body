from dataset import HubDataset
from model import get_model
from engine import train, eval
from transforms import data_transforms
import config
from arguments import parse_args

import pandas as pd
import torch
import argparse

scores = []

df = pd.read_csv("../input/hubmap-folds/train_256x256_5folds.csv")

for fold in range(config.FOLDS):

    best_metric = 0
    model = get_model()

    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    df_train = df_train.drop(columns = 'fold')
    df_valid = df_valid.drop(columns = 'fold')

    train_images = df_train.image_path.values.tolist()
    valid_images = df_valid.image_path.values.tolist()


    train_masks = df_train.mask_path.values
    valid_masks = df_valid.mask_path.values

    train_dataset = HubDataset(image_paths=train_images,mask_paths=train_masks,transforms=data_transforms["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True,pin_memory=True) 
    valid_dataset = HubDataset(image_paths=valid_images, mask_paths=valid_masks,transforms=data_transforms["valid"])
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.VALID_BATCH_SIZE,shuffle=False,pin_memory=True) 

    optimizer = torch.optim.AdamW(model.parameters(),lr=config.LR,amsgrad=True )

    print(f'============================== FOLD -- {fold} ==============================')
    for epoch in range(config.EPOCHS):
        print(f'==================== Epoch -- {epoch} ====================')
        train(model=model,train_loader=train_loader,device=config.DEVICE,optimizer=optimizer)

        dice_metric = eval(model=model,valid_loader=valid_loader,device=config.DEVICE,optimizer=optimizer)

        print(f'DICE Metric={dice_metric}')

        if best_metric <= dice_metric:
            best_metric = dice_metric
            torch.save(model.state_dict(),f'model-epoch-{epoch}'+str(fold)+'.pth')

# if __name__ == "__main__":

#     args = parse_args()