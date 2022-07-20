import config
from model import HubmapModel
from dataset import HubDataset
from augmentation import data_transforms
from engine import train, evaluation

import torch
import pandas as pd

## PRINT CONFIG ##
print("configuration :")
print(f" -- FOLDS : {config.FOLDS}")
print(f" -- MODEL : {config.MODEL}")
print(f" -- LR : {config.LR}")
print(f" -- TRAIN_BATCH_SIZE  : {config.TRAIN_BATCH_SIZE}")
print(f" -- VALID_BATCH_SIZE  : {config.VALID_BATCH_SIZE}")
print(f" -- EPOCHS  : {config.EPOCHS}")
print(f" -- CSV_PATH  : {config.CSV_PATH}")


df = pd.read_csv(config.CSV_PATH)

print(f"read csv- - - {config.CSV_PATH}")

for fold in {config.FOLDS}:

    best_metric = 0.0

    model = HubmapModel()
    model.to("cuda")

    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    df_train = df_train.drop(columns="fold")
    df_valid = df_valid.drop(columns="fold")

    train_images = df_train.image_path.values.tolist()
    valid_images = df_valid.image_path.values.tolist()

    train_masks = df_train.mask_path.values
    valid_masks = df_valid.mask_path.values

    train_dataset = HubDataset(
        image_paths=train_images,
        mask_paths=train_masks,
        transforms=data_transforms["train"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True
    )
    valid_dataset = HubDataset(
        image_paths=valid_images,
        mask_paths=valid_masks,
        transforms=data_transforms["valid"],
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.85, verbose=True
    )

    print(
        f"============================== FOLD -- {fold} =============================="
    )

    for epoch in range(config.EPOCHS):
        print(f"==================== Epoch -- {epoch} ====================")
        train(
            model=model,
            train_loader=train_loader,
            device=config.DEVICE,
            optimizer=optimizer,
        )
        scheduler.step()
        dice_score = evaluation(
            model=model,
            valid_loader=valid_loader,
            device=config.DEVICE,
            optimizer=optimizer,
        )

        print(f"DICE Metric={dice_score}")

        torch.save(model.state_dict(), "model-" + str(fold) + f"epoch-{epoch}" + ".pth")
