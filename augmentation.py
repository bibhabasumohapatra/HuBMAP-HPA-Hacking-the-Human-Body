import albumentations as A
import config

data_transforms = {
    "train": A.Compose([
        A.Resize(config.IMAGE_SIZE,config.IMAGE_SIZE),
            A.Normalize(
        mean=config.MEAN,
        std=config.STD,
        max_pixel_value=255.0,
        p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        ], p=1.0),
    
    "valid": A.Compose([
        A.Resize(config.IMAGE_SIZE,config.IMAGE_SIZE),
        A.Normalize(
        mean=config.MEAN,
        std=config.STD,
        max_pixel_value=255.0,
        p=1.0,
    ),    
    ], p=1.0)
}