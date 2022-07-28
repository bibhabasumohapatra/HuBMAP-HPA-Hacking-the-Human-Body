import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from common  import *
from lib.net.lookahead import *
from model import *
from dataset import *

import torch.cuda.amp as amp
is_amp = True  #True #False




#################################################################################################

def do_valid(net, valid_loader):

	valid_num = 0
	valid_probability = []
	valid_mask = []
	valid_loss = 0

	net = net.eval()
	start_timer = timer()
	for t, batch in enumerate(valid_loader):
		
		net.output_type = ['loss', 'inference']
		with torch.no_grad():
			with amp.autocast(enabled = is_amp):
				
				batch_size = len(batch['index'])
				batch['image'] = batch['image'].cuda()
				batch['mask' ] = batch['mask' ].cuda()
				batch['organ'] = batch['organ'].cuda()
				
				output = data_parallel(net, batch) #net(input)#
				loss0  = output['bce_loss'].mean()
			 
		valid_probability.append(output['probability'].data.cpu().numpy())
		valid_mask.append(batch['mask'].data.cpu().numpy())
		valid_num += batch_size
		valid_loss += batch_size*loss0.item()

		#debug
		if 0 :
			pass
			organ = batch['organ'].data.cpu().numpy()
			image = batch['image']
			mask  = batch['mask']
			probability  = output['probability']

			for b in range(batch_size):
				m = tensor_to_image(image[b])
				t = tensor_to_mask(mask[b,0])
				p = tensor_to_mask(probability[b,0])
				overlay = result_to_overlay(m, t, p )
				
				text = label_to_organ[organ[b]]
				draw_shadow_text(overlay,text,(5,15),0.7,(1,1,1),1)
				
				image_show_norm('overlay',overlay,min=0,max=1,resize=1)
				cv2.waitKey(0)


		#---
		print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)
		#if valid_num==200*4: break

	assert(valid_num == len(valid_loader.dataset))
	#print('')
	#------
	probability = np.concatenate(valid_probability)
	mask = np.concatenate(valid_mask)
	
	loss = valid_loss/valid_num   #np_binary_cross_entropy_loss(probability, mask)
	dice = compute_dice_score(probability, mask)
 
	dice = dice.mean()
	return [dice, loss,  0, 0]



 ##----------------

def run_train():

	fold = 0

	out_dir = root_dir + '/result/run20/segformer-mit-b2-aux5-768/fold-%d' % (fold)
	initial_checkpoint = None # out_dir + '/checkpoint/00001925.model.pth'  #
	#None #
 
	start_lr   = 5e-5 #0.0001
	batch_size = 6 #32 #32
	

	## setup  ----------------------------------------
	for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
	#backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

	log = Logger()
	log.open(out_dir+'/log.train.txt',mode='a')
	log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
	log.write('\t%s\n' % COMMON_STRING)
	log.write('\t__file__ = %s\n' % __file__)
	log.write('\tout_dir  = %s\n' % out_dir)
	log.write('\n')


	## dataset ----------------------------------------
	log.write('** dataset setting **\n')

	train_df, valid_df = make_fold(fold)
	# valid_df.to_csv('valid_df.fold%d.csv'%fold,index=False)
	# exit(0)

	train_dataset = HubmapDataset(train_df, train_augment5b)
	valid_dataset = HubmapDataset(valid_df, valid_augment5)
	
	train_loader  = DataLoader(
		train_dataset,
		sampler = RandomSampler(train_dataset),
		batch_size  = batch_size,
		drop_last   = True,
		num_workers = 8,
		pin_memory  = False,
		worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn = null_collate,
	)
 
	valid_loader = DataLoader(
		valid_dataset,
		sampler = SequentialSampler(valid_dataset),
		batch_size  = 8,
		drop_last   = False,
		num_workers = 4,
		pin_memory  = False,
		collate_fn = null_collate,
	)


	log.write('fold = %s\n'%str(fold))
	log.write('train_dataset : \n%s\n'%(train_dataset))
	log.write('valid_dataset : \n%s\n'%(valid_dataset))
	log.write('\n')

	## net ----------------------------------------
	log.write('** net setting **\n')
	
	scaler = amp.GradScaler(enabled = is_amp)
	net = Net().cuda()

	if initial_checkpoint is not None:
		f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
		start_iteration = f['iteration']
		start_epoch = f['epoch']
		state_dict  = f['state_dict']
		net.load_state_dict(state_dict,strict=False)  #True
	else:
		start_iteration = 0
		start_epoch = 0
		net.load_pretrain()


	log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
	log.write('\n')


	## optimiser ----------------------------------
	if 0: ##freeze
		for p in net.stem.parameters():   p.requires_grad = False
		pass

	def freeze_bn(net):
		for m in net.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()
				m.weight.requires_grad = False
				m.bias.requires_grad = False
	#freeze_bn(net)

	#-----------------------------------------------

	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
	#optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr), alpha=0.5, k=5)
	
	log.write('optimizer\n  %s\n'%(optimizer))
	log.write('\n')
	
	num_iteration = 1000*len(train_loader)
	iter_log   = len(train_loader)*3 #479
	iter_valid = iter_log
	iter_save  = iter_log
 
	## start training here! ##############################################
	#array([0.57142857, 0.42857143])
	log.write('** start training here! **\n')
	log.write('   batch_size = %d \n'%(batch_size))
	log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
	log.write('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
	log.write('rate     iter  epoch | dice   loss   tp     tn     | loss           | time           \n')
	log.write('-------------------------------------------------------------------------------------\n')
			  #0.00100   0.50  0.80 | 0.891  0.020  0.000  0.000  | 0.000  0.000   |  0 hr 02 min
	
	def message(mode='print'):
		asterisk = ' '
		if mode==('print'):
			loss = batch_loss
		if mode==('log'):
			loss = train_loss
			if (iteration % iter_save == 0): asterisk = '*'
		
		text = \
			('%0.2e   %08d%s %6.2f | '%(rate, iteration, asterisk, epoch,)).replace('e-0','e-').replace('e+0','e+') + \
			'%4.3f  %4.3f  %4.4f  %4.3f   | '%(*valid_loss,) + \
			'%4.3f  %4.3f   | '%(*loss,) + \
			'%s' % (time_to_str(timer() - start_timer,'min'))
		
		return text

	#----
	valid_loss = np.zeros(4,np.float32)
	train_loss = np.zeros(2,np.float32)
	batch_loss = np.zeros_like(train_loss)
	sum_train_loss = np.zeros_like(train_loss)
	sum_train = 0
	



	start_timer = timer()
	iteration = start_iteration
	epoch = start_epoch
	rate = 0
	while iteration < num_iteration:
		for t, batch in enumerate(train_loader):
			
			if iteration%iter_save==0:
				if iteration != start_iteration:
					torch.save({
						'state_dict': net.state_dict(),
						'iteration': iteration,
						'epoch': epoch,
					}, out_dir + '/checkpoint/%08d.model.pth' %  (iteration))
					pass
			
			
			if (iteration%iter_valid==0): # or (t==len(train_loader)-1):
				#if iteration!=start_iteration:
				valid_loss = do_valid(net, valid_loader)  #
				pass
			
			
			if (iteration%iter_log==0) or (iteration%iter_valid==0):
				print('\r', end='', flush=True)
				log.write(message(mode='log') + '\n')
				
				
			# learning rate schduler ------------
			# adjust_learning_rate(optimizer, scheduler(epoch))
			rate = get_learning_rate(optimizer)[0] #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)
			
			# one iteration update  -------------
			batch_size = len(batch['index'])
			batch['image'] = batch['image'].half().cuda()
			batch['mask' ] = batch['mask' ].half().cuda()
			batch['organ'] = batch['organ'].cuda()
			
			
			net.train()
			net.output_type = ['loss']
			#with torch.autograd.set_detect_anomaly(True):
			if 1:
				with amp.autocast(enabled = is_amp):
					output = data_parallel(net,batch)
					loss0  = output['bce_loss'].mean()
					loss1  = output['aux2_loss'].mean()
				#loss1  = output['lovasz_loss'].mean()
			
				optimizer.zero_grad()
				scaler.scale(loss0+0.2*loss1).backward()
				
				scaler.unscale_(optimizer)
				#torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
				scaler.step(optimizer)
				scaler.update()
			
			
			# print statistics  --------
			batch_loss[:2] = [loss0.item(),loss1.item()]
			sum_train_loss += batch_loss
			sum_train += 1
			if t % 100 == 0:
				train_loss = sum_train_loss / (sum_train + 1e-12)
				sum_train_loss[...] = 0
				sum_train = 0
			
			print('\r', end='', flush=True)
			print(message(mode='print'), end='', flush=True)
			epoch += 1 / len(train_loader)
			iteration += 1
			
			# debug  --------
			#if 1:
			# if (iteration%5==0):
		 
				
		torch.cuda.empty_cache()
	log.write('\n')

# main #################################################################
if __name__ == '__main__':
	run_train()

'''
 

'''
