import cv2
import torch
import numpy as np
import tifffile as tiff

from skimage import io, transform, filters 

class HubDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, pixel_size=None, augmentations=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.augmentations = augmentations
        self.pixel_size = pixel_size
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self,item):
        
        image = tiff.imread(self.image_path[item])
        mask = io.imread(self.mask_path[item])
        mask = mask.reshape(mask.shape[0],mask.shape[1],1)
        
	image = image.astype(np.float32)/255
	mask  = mask.astype(np.float32)/255
		
        
		# s = self.pixel_size/0.4 * (config.IMAGE_SIZE/image.shape[0])
        ## resize
        #image = cv2.resize(image,dsize=None, fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
		#mask  = cv2.resize(mask, dsize=None, fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
        image = cv2.resize(image, dsize=(config.IMAGE_SIZE, config.IMAGE_SIZE),interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(config.IMAGE_SIZE, config.IMAGE_SIZE),interpolation=cv2.INTER_LINEAR)
        
        # image = image - np.min(image)
        # image = image / np.max(image)
        
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)

    
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "mask" : torch.tensor(mask, dtype=torch.float),
        }
