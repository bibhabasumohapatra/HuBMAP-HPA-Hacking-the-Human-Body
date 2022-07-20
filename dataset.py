import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from skimage import io


class HubDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = io.imread(self.image_paths[item])

        if self.mask_paths is not None:
            mask = io.imread(self.mask_paths[item])
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

            if self.transforms is not None:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            image = np.transpose(image, (2, 0, 1))
            mask = np.transpose(mask, (2, 0, 1))
            return {
                "image": torch.tensor(image, dtype=float),
                "mask": torch.tensor(mask, dtype=float),
            }
        else:
            if self.transforms is not None:
                augmented = self.transforms(
                    image=image,
                )
                image = augmented["image"]

            image = np.transpose(image, (2, 0, 1))

            return {
                "image": torch.tensor(image, dtype=float),
            }
