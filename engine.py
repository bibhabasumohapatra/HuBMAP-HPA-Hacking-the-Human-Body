import torch
import torch.nn as nn

import torch
import torch.nn as nn


import config
import loss as losses

criterion = losses.criterion

scaler = torch.cuda.amp.GradScaler()


def train(model, train_loader, device, optimizer):
    model.train()
    running_train_loss = 0.0
    for data in train_loader:
        inputs = data["image"]
        masks = data["mask"]

        ## CUDA
        inputs = inputs.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        ## forward
        with torch.cuda.amp.autocast():
            outputs = model(
                inputs,
            )
            loss = criterion(outputs, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()

    train_loss_value = running_train_loss / len(train_loader)
    print(f"train custom Loss is {train_loss_value}")


@torch.no_grad()
def evaluation(model, valid_loader, device):
    model.eval()
    running_dice_score = 0.0
    running_val_loss = 0.0
    for data in valid_loader:
        inputs = data["image"]
        masks = data["mask"]

        inputs = inputs.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        output = model(
            inputs,
        )
        running_val_loss += criterion(output, masks)

        output = torch.sigmoid(output)
        running_dice_score += losses.dice_metric(masks, output)

    val_loss = running_val_loss / len(valid_loader)
    dice_score = running_dice_score / len(valid_loader)

    print(f"valid custom Loss is {val_loss}")

    return dice_score
