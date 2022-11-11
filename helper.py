import torch
import numpy as np
import tifffile as tiff
from torch import nn

class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def rle_encode(mask):
	m = mask.T.flatten()
	m = np.concatenate([[0], m, [0]])
	run = np.where(m[1:] != m[:-1])[0] + 1
	run[1::2] -= run[::2]
	rle =  ' '.join(str(r) for r in run)
	return rle

def read_tiff(image_file, mode='rgb'):
	image = tiff.imread(image_file)
	image = image.squeeze()
	if image.shape[0] == 3:
		image = image.transpose(1, 2, 0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	return image


organ_meta = dotdict(
	kidney = dotdict(
		label = 1,
		um    = 0.5000,
		ftu   ='glomeruli',
	),
	prostate = dotdict(
		label = 2,
		um    = 6.2630,
		ftu   ='glandular acinus',
	),
	largeintestine = dotdict(
		label = 3,
		um    = 0.2290,
		ftu   ='crypt',
	),
	spleen = dotdict(
		label = 4,
		um    = 0.4945,
		ftu   ='white pulp',
	),
	lung = dotdict(
		label = 5,
		um    = 0.7562,
		ftu   ='alveolus',
	),
)

organ_to_label = {k: organ_meta[k].label for k in organ_meta.keys()}
label_to_organ = {v:k for k,v in organ_to_label.items()}

def image_to_tensor(image, mode='rgb'):
    if  mode=='bgr' :
        image = image[:,:,::-1]
    
    x = image.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x)
    return x

