from common import *

# flip ##----

def do_random_flip(image, mask):
    if np.random.rand()>0.5:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)
    if np.random.rand()>0.5:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)
    if np.random.rand()>0.5:
        image = image.transpose(1,0,2)
        mask = mask.transpose(1,0)
    
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    return image, mask

def do_random_rot90(image, mask):
    r = np.random.choice([
        0,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ])
    if r==0:
        return image, mask
    else:
        image = cv2.rotate(image, r)
        mask = cv2.rotate(mask, r)
        return image, mask


# crop ##----
def do_crop(image, mask, size, xy=(0,0)):
	height, width = image.shape[:2]
	x, y = xy
	if x is None: x = (width-size)//2
	if y is None: y = (height-size)//2
	
	image = image[y:y+size,x:x+size]
	mask  = mask[y:y+size,x:x+size]
	return image, mask


def do_random_crop(image, mask, size):
    height, width = image.shape[:2]
    x = np.random.choice(width -size) if width>size else 0
    y = np.random.choice(height-size) if height>size else 0
    image = image[y:y+size,x:x+size]
    mask  = mask[y:y+size,x:x+size]
    return image, mask

# transform ##----
def do_random_rotate_scale(image, mask, angle=30, scale=[0.8,1.2] ):
    angle = np.random.uniform(-angle, angle)
    scale = np.random.uniform(*scale) if scale is not None else 1
    
    height, width = image.shape[:2]
    center = (height // 2, width // 2)
    
    transform = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine( image, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask  = cv2.warpAffine( mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask




#noise
def do_random_noise(image, mask, mag=0.1):
	height, width = image.shape[:2]
	noise = np.random.uniform(-1,1, (height, width,1))*mag
	image = image + noise
	image = np.clip(image,0,1)
	return image, mask

# https://openreview.net/pdf?id=rkBBChjiG
# <todo> mixup/cutout

#intensity
def do_random_contast(image, mask, mag=0.3):
	alpha = 1 + random.uniform(-1,1)*mag
	image = image * alpha
	image = np.clip(image,0,1)
	return image, mask


def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
	image = (image*255).astype(np.uint8)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	
	h = hsv[:, :, 0].astype(np.float32)  # hue
	s = hsv[:, :, 1].astype(np.float32)  # saturation
	v = hsv[:, :, 2].astype(np.float32)  # value
	h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
	s =  s*(1 + random.uniform(-1,1)*mag[1])
	v =  v*(1 + random.uniform(-1,1)*mag[2])
	
	hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
	hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
	hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
	image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	image = image.astype(np.float32)/255
	return image, mask


def do_gray(image, mask):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	return image, mask