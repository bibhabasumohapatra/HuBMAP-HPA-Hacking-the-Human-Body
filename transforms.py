import albumentations as A
import cv2
import config

data_transforms = {
    "train": A.Compose([
        A.Resize(config.IMAGE_SIZE,config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST),
            A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
# #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.CoarseDropout(max_holes=8, max_height=config.IMAGE_SIZE//20, max_width=config.IMAGE_SIZE//20,
                        min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),
    
    "valid": A.Compose([
        A.Resize(config.IMAGE_SIZE,config.IMAGE_SIZE, interpolation=cv2.INTER_NEAREST),
        A.Normalize(
        mean=[0.7720342,  0.74582646, 0.76392896],
        std=[0.24745085, 0.26182273, 0.25782376],
        max_pixel_value=255.0,
        p=1.0,
    ),    
    ], p=1.0)
}
