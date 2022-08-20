from kaggle_hubmap_kv3 import *
from daformer import *
from pvt_v2 import *


image_size=768
#################################################################

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


class Net(nn.Module):
	
 
	def __init__(self,
	             encoder=pvt_v2_b4,
	             decoder=daformer_conv3x3,
	             encoder_cfg=dict(img_size=image_size),
	             decoder_cfg=dict(),
	             ):
		super(Net, self).__init__()
		decoder_dim = decoder_cfg.get('decoder_dim', 320)
		
		# ----
		
		self.rgb = RGB()
		
		self.encoder = encoder(
			**encoder_cfg
		)
		encoder_dim = self.encoder.embed_dims
		# [64, 128, 320, 512]
		
		self.decoder = decoder(
			encoder_dim=encoder_dim,
			decoder_dim=decoder_dim,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(decoder_dim, 1, kernel_size=1),
		)
 
	
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B, C, H, W = x.shape
		encoder = self.encoder(x)
		#print([f.shape for f in encoder])
		
		last, decoder = self.decoder(encoder)
		logit = self.logit(last)
		# print(logit.shape)
		
		output = {}
		probability_from_logit = torch.sigmoid(logit)
		output['probability'] = probability_from_logit
		
		return output
 
 


def run_check_net():
	batch_size = 2
	#image_size = image_size #800
	
	# ---
	batch = {
		'image': torch.from_numpy(np.random.uniform(-1, 1, (batch_size, 3, image_size, image_size))).float(),
		'mask': torch.from_numpy(np.random.choice(2, (batch_size, 1, image_size, image_size))).float(),
		'organ': torch.from_numpy(np.random.choice(5, (batch_size, 1))).long(),
	}
	batch = {k: v.cuda() for k, v in batch.items()}
	
	net = Net().cuda()
	
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k, v in batch.items():
		print('%32s :' % k, v.shape)
	
	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	# run_check_mit()
	run_check_net()
