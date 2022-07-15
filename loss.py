import segmentation_models_pytorch as smp

import torch
import numpy as np
import torch.nn as nn

import config

loss = config.LOSS


JaccardLoss = smp.losses.JaccardLoss(mode='binary',smooth=1.0)
DiceLoss = smp.losses.DiceLoss(mode='binary', smooth=1.0)
FocalLoss = smp.losses.FocalLoss(mode="binary")
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)

### Declaration

if loss == "Dice":
    def criterion(y_pred, y_true):
        return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

if loss == "BCE":
    def criterion(y_pred, y_true):
        return BCELoss(y_pred, y_true)

if loss == "BCE_Tversky":
    def criterion(y_pred, y_true):
        return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)

if loss == "Lovasz":
    def criterion(y_pred, y_true):
        return LovaszLoss(y_pred, y_true)

if loss == "Dice_BCE":
    def criterion(y_pred, y_true):
        return 0.5*BCELoss(y_pred, y_true) + 0.5*DiceLoss(y_pred, y_true)

if loss == "FocalLoss":
    def criterion(y_pred, y_true):
        return FocalLoss(y_pred, y_true) 

def dice_coef(_mask1,_mask2):
    
    batch_size = _mask1.shape[0]
    dice_total = 0.0
    
    for idx in range(batch_size):
        mask1 = _mask1[idx].reshape(_mask1[idx].shape[1],_mask1[idx].shape[2]).cpu().numpy()
        mask2 = _mask2[idx].reshape(_mask2[idx].shape[1],_mask2[idx].shape[2]).cpu().numpy()
        intersect = np.sum(mask1*mask2)
        fsum = np.sum(mask1)
        ssum = np.sum(mask2)
        eps = 1e-7 ##  for empty masks the numerator should not be divivded by zero
        dice = (2 * intersect + eps) / (fsum + ssum + eps)
        dice = np.mean(dice)
        
        dice_total += dice
        
    final_dice = dice_total/batch_size
    final_dice = round(final_dice, 4)  # for easy reading till 4 decimal places
    return final_dice  
