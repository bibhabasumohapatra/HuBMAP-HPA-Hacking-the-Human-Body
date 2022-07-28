import os
import torch
import pandas as pd

import config as config
from model import HubmapModel
from dataset import HubDataset
from augmentation import valid_augment5, train_augment5a
from engine import train, evaluation



## PRINT CONFIG ##
print("configuration :")
print(f" -- FOLDS : {config.FOLDS}")
print(f" -- MODEL : {config.MODEL_PATH}")
print(f" -- LR : {config.LR}")
print(f" -- TRAIN_BATCH_SIZE  : {config.TRAIN_BATCH_SIZE}")
print(f" -- VALID_BATCH_SIZE  : {config.VALID_BATCH_SIZE}")
print(f" -- EPOCHS  : {config.EPOCHS}")
print(f" -- CSV_PATH  : {config.CSV_PATH}")


df = pd.read_csv(config.CSV_PATH)

print(f"read csv- - - {config.CSV_PATH}")

for fold in {config.FOLDS}:

    best_score = 0.0

    model = HubmapModel()
    model.to("cuda")
    
    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    df_train = df_train.drop(columns = 'fold')
    df_valid = df_valid.drop(columns = 'fold')

    train_ids = df_train.id.values.tolist()
    valid_ids = df_valid.id.values.tolist()

    train_images = [os.path.join("../input/hubmap-organ-segmentation/train_images",str(i) + ".tiff") for i in train_ids]
    train_masks =  [os.path.join("../input/hubmap-hpa-2022-maskdataset/hubmap_2022_MaskDataset",str(i) + ".png") for i in train_ids]

    valid_images = [os.path.join("../input/hubmap-organ-segmentation/train_images",str(i) + ".tiff") for i in valid_ids]
    valid_masks =  [os.path.join("../input/hubmap-hpa-2022-maskdataset/hubmap_2022_MaskDataset",str(i) + ".png") for i in valid_ids]

    train_dataset = HubDataset(image_path=train_images,mask_path=train_masks,augmentations=train_augment5a)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.TRAIN_BATCH_SIZE,shuffle=True,pin_memory=True) 
    valid_dataset = HubDataset(image_path=valid_images, mask_path=valid_masks, augmentations=valid_augment5)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.VALID_BATCH_SIZE,shuffle=False,pin_memory=True) 

    optimizer = torch.optim.Adam(model.parameters(),lr=config.LR,)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, verbose=True)

    print(f'============================== FOLD -- {fold} ==============================')

    for epoch in range(config.EPOCHS):
        print(f'==================== Epoch -- {epoch} ====================')
        train(model=model,train_loader=train_loader,device=config.DEVICE,optimizer=optimizer)
        scheduler.step()
        dice_score = evaluation(model=model,valid_loader=valid_loader,device=config.DEVICE,optimizer=optimizer)

        print(f'validation DICE Metric={dice_score}')
        
        if dice_score > best_score:
            best_score = dice_score
            torch.save(model.state_dict(),'model-'+str(fold) +'.pth')
