from kaggle_hubmap_v2 import *
from common import *

from mit import *

#######################################################################################################
## https://github.com/lucidrains/segformer-pytorch/blob/main/segformer_pytorch/segformer_pytorch.py
# https://github.com/UAws/CV-3315-Is-All-You-Need
class MixUpSample(nn.Module):
	def __init__( self, scale_factor=2):
		super().__init__()
		self.mixing = nn.Parameter(torch.tensor(0.5))
		self.scale_factor = scale_factor
	
	def forward(self, x):
		x = self.mixing *F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
		    + (1-self.mixing )*F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
		return x
	
	
class SegformerDecoder(nn.Module):
	def __init__(
			self,
			encoder_dim = [32, 64, 160, 256],
			decoder_dim = 256,
	):
		super().__init__()
		self.mixing = nn.Parameter(torch.FloatTensor([0.5,0.5,0.5,0.5]))
		self.mlp = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(dim, decoder_dim, 1, padding= 0,  bias=False), #follow mmseg to use conv-bn-relu
				nn.BatchNorm2d(decoder_dim),
				nn.ReLU(inplace=True),
				MixUpSample(2**i) if i!=0 else nn.Identity(),
			) for i, dim in enumerate(encoder_dim)])
		
		self.fuse = nn.Sequential(
			nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
			nn.BatchNorm2d(decoder_dim),
			nn.ReLU(inplace=True),
			# nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
			# nn.BatchNorm2d(decoder_dim),
			# nn.ReLU(inplace=True),
		)
	
	def forward(self, feature):
		
		out = []
		for i,f in enumerate(feature):
			f = self.mlp[i](f)
			out.append(f)
		 
		x = self.fuse(torch.cat(out, dim = 1))
		return x, out


#<todo>
# do a reverse mit upsize + conv + Mix FFN

#################################################################

class RGB(nn.Module):
	IMAGE_RGB_MEAN = [0.485, 0.456, 0.406] #[0.5, 0.5, 0.5]
	IMAGE_RGB_STD  = [0.229, 0.224, 0.225] #[0.5, 0.5, 0.5]
	
	def __init__(self,):
		super(RGB, self).__init__()
		self.register_buffer('mean', torch.zeros(1,3,1,1))
		self.register_buffer('std', torch.ones(1,3,1,1))
		self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
		self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)
	
	def forward(self, x):
		x = (x-self.mean)/self.std
		return x


class Net(nn.Module):
	def load_pretrain( self,):
		checkpoint = cfg[self.arch]['checkpoint']
		print('load %s'%checkpoint)
		checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)  #True
		print(self.encoder.load_state_dict(checkpoint,strict=False))  #True
	
	
	def __init__( self,):
		super(Net, self).__init__()
		self.output_type = ['inference', 'loss']
		self.rgb = RGB()
		self.dropout = nn.Dropout(0.1)
		
		self.arch = 'mit_b2'
		self.encoder = cfg[self.arch]['builder']()
		encoder_dim = self.encoder.embed_dims
		#[64, 128, 320, 512]
		
		self.decoder = SegformerDecoder(
			encoder_dim = encoder_dim,
			decoder_dim = 320,
		)
		self.logit = nn.Sequential(
			nn.Conv2d(320, 1, kernel_size=1, padding=0),
		)
		self.aux = nn.ModuleList([
			nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(4)
		])
	
	
	def forward(self, batch):
		
		x = batch['image']
		x = self.rgb(x)
		
		B,C,H,W = x.shape
		encoder = self.encoder(x)
		#print([f.shape for f in encoder])
		
		last, decoder = self.decoder(encoder)
		last  = self.dropout(last)
		logit = self.logit(last)
		logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
		#print(logit.shape)
		
		output = {}
		if 'loss' in self.output_type:
			output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
			for i in range(4):
				output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](encoder[i]),batch['mask'])
		 
		if 'inference' in self.output_type:
			output['probability'] = torch.sigmoid(logit)
		
		return output

def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit,mask)
	return loss

'''

x.shape
torch.Size([4, 3, 320, 320])


print([f.shape for f in feature])
[
torch.Size([4, 128, 80, 80]),
torch.Size([4, 256, 40, 40]),
torch.Size([4, 512, 20, 20]),
torch.Size([4, 1024, 10, 10])
]


'''
def run_check_net():
	batch_size = 2
	image_size = 768
	
	#---
	batch = {
		'image' : torch.from_numpy( np.random.uniform(-1,1,(batch_size,3,image_size,image_size)) ).float(),
		'mask'  : torch.from_numpy( np.random.choice(2,(batch_size,1,image_size,image_size)) ).float(),
		'organ' : torch.from_numpy( np.random.choice(5,(batch_size)) ).long(),
	}
	batch = {k:v.cuda() for k,v in batch.items()}
	
	
	
	net = Net().cuda()
	# torch.save({ 'state_dict': net.state_dict() },  'model.pth' )
	# exit(0)
	net.load_pretrain()
	
	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)
	
	print('batch')
	for k,v in batch.items():
		print('%32s :'%k, v.shape)
	
	print('output')
	for k,v in output.items():
		if 'loss' not in k:
			print('%32s :'%k, v.shape)
	for k,v in output.items():
		if 'loss' in k:
			print('%32s :'%k, v.item())

'''
class mit_b0(MixVisionTransformer):
	def __init__(self, **kwargs):
		super(mit_b0, self).__init__(
			patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
			qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
			drop_rate=0.0, drop_path_rate=0.1)


'''
def run_check_mit():
	batch_size = 4
	image_size = 768
	
	checkpoint = '/root/Downloads/mit_b0.pth'
	#---
	image = torch.from_numpy( np.random.uniform(-1,1,(batch_size,3,image_size,image_size)) ).float()
	image = image.cuda()
	
	net = mit_b0()
	
	state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
	net.load_state_dict(state_dict,strict=False)  #True
	#Unexpected key(s) in state_dict: "head.weight", "head.bias".
	
	net = net.cuda()
	feature = net(image)
	print([f.shape for f in feature])

# main #################################################################
if __name__ == '__main__':
	#run_check_mit()
	run_check_net()
