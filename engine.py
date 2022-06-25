from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch
import torch.nn as nn


dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

def train(model,train_loader,device,optimizer):
    model.train()
    running_train_loss = 0.0
    for data in train_loader:
        inputs = data['image']
        masks = data['mask']

        inputs = inputs.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs,)
        loss = DiceLoss(sigmoid=True)(outputs, masks)
        loss.backward()
        optimizer.step()
        running_train_loss +=loss.item()
        
    train_loss_value = running_train_loss/len(train_loader)
    print(f'train DICE loss is {train_loss_value}')
    
def eval(model,valid_loader,device,optimizer):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            inputs = data['image']
            masks = data['mask']
            
            inputs = inputs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            output = model(inputs,)
            running_val_loss +=  DiceLoss(sigmoid=True)(output, masks)
            dice_metric(y_pred=output, y=masks)

        val_loss = running_val_loss/len(valid_loader)    
        print(f'valid DICE loss is {val_loss}')
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    return metric 