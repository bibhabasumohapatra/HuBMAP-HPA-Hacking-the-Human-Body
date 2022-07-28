from kaggle_hubmap_v2 import *
from common import *
from sklearn.model_selection import KFold

from augmentation import *


image_size = 768  #672 #736 #768 #32*24



#------------------------------
def make_fold(fold=0):
	df = pd.read_csv(root_dir + '/data/hubmap-organ-segmentation/train.csv')
	
	num_fold = 5
	skf = KFold(n_splits=num_fold, shuffle=True,random_state=42)
	
	df.loc[:,'fold']=-1
	for f,(t_idx, v_idx) in enumerate(skf.split(X=df['id'], y=df['organ'])):
		df.iloc[v_idx,-1]=f
	
	#check
	if 0:
		for f in range(num_fold):
			train_df=df[df.fold!=f].reset_index(drop=True)
			valid_df=df[df.fold==f].reset_index(drop=True)
			
			print('fold %d'%f)
			t = train_df.organ.value_counts().to_dict()
			v = valid_df.organ.value_counts().to_dict()
			for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
				print('%32s %3d (%0.3f)  %3d (%0.3f)'%(k,t.get(k,0),t.get(k,0)/len(train_df),v.get(k,0),v.get(k,0)/len(valid_df)))
			
			print('')
			zz=0
	
	train_df=df[df.fold!=fold].reset_index(drop=True)
	valid_df=df[df.fold==fold].reset_index(drop=True)
	return train_df,valid_df


def pad_to_multiple(image, mask, multiple=32, min_size=image_size):
	
	sh,sw,_ = image.shape
	ph = max(min_size,int(np.ceil(sh/32))*32) -sh
	pw = max(min_size,int(np.ceil(sw/32))*32) -sw
 
	image = np.pad(image, ((0,ph), (0,pw), (0,0)), 'constant', constant_values=0)
	mask  = np.pad(mask, ((0,ph), (0,pw)), 'constant', constant_values=0)
	return image, mask
	

####################################################################################################

class HubmapDataset(Dataset):
	def __init__(self, df, augment=None):
		
		self.df = df
		self.augment = augment
		self.length = len(self.df)
	
	def __str__(self):
		string = ''
		string += '\tlen = %d\n' % len(self)
		
		d = self.df.organ.value_counts().to_dict()
		for k in ['kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
			string +=  '%24s %3d (%0.3f) \n'%(k,d.get(k,0),d.get(k,0)/len(self.df))
		return string
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, index):
		d = self.df.iloc[index]
		id = d['id']
		organ = organ_to_label[d.organ]
		
		image = cv2.imread(root_dir + '/data/hubmap-organ-segmentation/train_images_png/%d.png'%id, cv2.IMREAD_COLOR)
		mask  = cv2.imread(root_dir + '/data/hubmap-organ-segmentation/train_annotations_png/%d.png'%id, cv2.IMREAD_GRAYSCALE)
		image = image.astype(np.float32)/255
		mask  = mask.astype(np.float32)/255
		
		s = d.pixel_size/0.4 * (image_size/3000)
		#image = cv2.resize(image,dsize=None, fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
		#mask  = cv2.resize(mask, dsize=None, fx=s,fy=s,interpolation=cv2.INTER_LINEAR)
		image = cv2.resize(image,dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
		mask  = cv2.resize(mask, dsize=(image_size,image_size),interpolation=cv2.INTER_LINEAR)
		
		if self.augment is not None:
			image, mask = self.augment(image, mask, organ)
		
		 
		r ={}
		r['index']= index
		r['id'] = id
		r['organ'] = torch.tensor([organ], dtype=torch.long)
		r['image'] = image_to_tensor(image)
		r['mask' ] = mask_to_tensor(mask>0.5)
		return r

tensor_list = [
	'mask', 'image', 'organ',
]


def null_collate(batch):
	d = {}
	key = batch[0].keys()
	for k in key:
		v = [b[k] for b in batch]
		if k in tensor_list:
			v = torch.stack(v)
		d[k] = v
	
	d['mask'] = d['mask'].unsqueeze(1)
	d['organ'] = d['organ'].reshape(-1)
	return d

##############################################################################################################

def image_to_tensor(image, mode='bgr'): #image mode
	if mode=='bgr':
		image = image[:,:,::-1]
	x = image
	x = x.transpose(2,0,1)
	x = np.ascontiguousarray(x)
	x = torch.tensor(x, dtype=torch.float)
	return x

def tensor_to_image(x, mode='bgr'):
	image = x.data.cpu().numpy()
	image = image.transpose(1,2,0)
	if mode=='bgr':
		image = image[:,:,::-1]
	image = np.ascontiguousarray(image)
	image = image.astype(np.float32)
	return image

def mask_to_tensor(mask):
	x = mask
	x = torch.tensor(x, dtype=torch.float)
	return x

def tensor_to_mask(x):
	mask = x.data.cpu().numpy()
	mask = mask.astype(np.float32)
	return mask






########################################################################
def valid_augment5(image, mask, organ):
	#image, mask  = do_crop(image, mask, image_size, xy=(None,None))
	return image, mask


def train_augment5a(image, mask, organ):
	
	image, mask = do_random_flip(image, mask)
	image, mask = do_random_rot90(image, mask)
	
	for fn in np.random.choice([
		lambda image, mask : (image, mask),
		lambda image, mask : do_random_noise(image, mask, mag=0.1),
		lambda image, mask : do_random_contast(image, mask, mag=0.25),
		lambda image, mask : do_random_hsv(image, mask, mag=[0.30,0.30,0])
	],2): image, mask =  fn(image, mask)
 
	for fn in np.random.choice([
		lambda image, mask : (image, mask),
		lambda image, mask : do_random_rotate_scale(image, mask, angle=45,scale=[0.5,2]),
	],1): image, mask = fn(image, mask)
	
	return image, mask



def train_augment5b(image, mask, organ):
	image, mask = do_random_flip(image, mask)
	image, mask = do_random_rot90(image, mask)
	
	for fn in np.random.choice([
		lambda image, mask: (image, mask),
		lambda image, mask: do_random_noise(image, mask, mag=0.1),
		lambda image, mask: do_random_contast(image, mask, mag=0.40),
		lambda image, mask: do_random_hsv(image, mask, mag=[0.40, 0.40, 0])
	], 2): image, mask = fn(image, mask)
	
	for fn in np.random.choice([
		lambda image, mask: (image, mask),
		lambda image, mask: do_random_rotate_scale(image, mask, angle=45, scale=[0.50, 2.0]),
	], 1): image, mask = fn(image, mask)
	
	return image, mask

########################################################################
def run_check_dataset():
 
	
	fold=0
	train_df, valid_df = make_fold(fold)
	
	
	dataset = HubmapDataset(valid_df)
	print(dataset)
	
	for i in range(5):
		r = dataset[i]
		print(r['index'], 'id = ', r['id'],'-----------')
		for k in tensor_list:
			v = r[k]
			print(k)
			print('\t',v.shape, v.is_contiguous(), v.min(), v.max())
			print('\t',v.reshape(-1)[:8], '...')
			print('\t',v.reshape(-1)[-8:])
		print('')
	
	
	loader = DataLoader(
		dataset,
		sampler = SequentialSampler(dataset),
		batch_size  = 8,
		drop_last   = True,
		num_workers = 0,
		pin_memory  = False,
		worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn = null_collate,
	)
	print(loader.batch_size,len(loader),len(dataset))
	print('')
	
	for t, batch in enumerate(loader):
		if t>5: break
		print('batch ', t,'===================')
		print('index', batch['index'])
		for k in tensor_list:
			v = batch[k]
			print(k)
			print('\t',v.shape, v.is_contiguous())
			print('\t',v.reshape(-1)[:8])
		print('')



def run_check_augment():
	def augment(image, mask, organ):
		#image, mask = do_random_flip(image, mask)
		#image, mask = do_random_rot90(image, mask)
		#image, mask = do_random_rotate_scale(image, mask, angle=45,scale=0.25)
		
		#image, mask = do_random_noise(image, mask, mag=0.1)
		#image, mask = do_random_contast(image, mask, mag=0.2)
		image, mask = do_random_hsv(image, mask, mag=[0.30,0.30,0.25])
		#image, mask = do_gray(image, mask)
		#image, mask = train_augment(image, mask, organ)
		return image, mask
	
	#----
	
	fold=0
	train_df, valid_df = make_fold(fold)
	dataset = HubmapDataset( train_df)
	print(dataset)
	
	for i in range(1000):
		#for i in np.random.choice(len(dataset),100):
		r = dataset[i]
		image = tensor_to_image(r['image'])
		mask  = tensor_to_mask(r['mask'])
		organ  = r['organ']
		overlay = draw_contour_overlay(image.copy(), mask, color=(0,0,255), thickness=3)
		
		print('%2d --------------------------- '%(i))
		#image_show('overlay', overlay)
		#image_show('mask', mask)
		#image_show('image', image)
		image_show('all', np.hstack([image,overlay,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)]))
		cv2.waitKey(1)
		
		if 1:
			for i in range(100):
				image1, mask1 =  augment(image.copy(), mask.copy(), organ)
				#image1, mask1 = do_random_crop(image1, mask1, crop_size)
		
				overlay1 = draw_contour_overlay(image1.copy(), mask1, color=(0,0,255), thickness=3)
		
				image_show('all1', np.hstack([image1,overlay1,cv2.cvtColor(mask1,cv2.COLOR_GRAY2BGR)]))
				cv2.waitKey(0)
				
# main #################################################################
if __name__ == '__main__':
	
	run_check_dataset()
	#run_check_augment()
