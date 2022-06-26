from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from metrics import dice_coef
# dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

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
    running_dice_score = 0.0
    running_val_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            inputs = data['image']
            masks = data['mask']
            
            inputs = inputs.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            output = model(inputs,)
            running_val_loss +=  DiceLoss(sigmoid=True)(output, masks)
            
            output = torch.sigmoid(output)
            running_dice_score += dice_coef(masks, output)
 
        val_loss = running_val_loss/len(valid_loader) 
        dice_score = running_dice_score/len(valid_loader)
        
        print(f'valid DICE loss is {val_loss}')
        
    return dice_score
