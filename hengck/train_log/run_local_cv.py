import os

import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from common import *
from model import *
from dataset import *

import torch.cuda.amp as amp

is_amp = True  # True #False

organ_threshold = {
	'kidney': 0.50,
	'prostate': 0.50,
	'largeintestine': 0.50,
	'spleen': 0.50,
	'lung': 0.20,
}


#################################################################################################
# Stochastic Weight Averaging
# https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
def do_swa(checkpoint):
	skip = ['relative_position_index', 'num_batches_tracked']
	
	K = len(checkpoint)
	swa = None
	
	for k in range(K):
		state_dict = torch.load(checkpoint[k], map_location=lambda storage, loc: storage)['state_dict']
		if swa is None:
			swa = state_dict
		else:
			for k, v in state_dict.items():
				# print(k)
				if any(s in k for s in skip): continue
				swa[k] += v
	
	for k, v in swa.items():
		if any(s in k for s in skip): continue
		swa[k] /= K
	
	return swa


def run_submit():
	out_dir = root_dir + '/result/run20/upernet-swin_small_patch4_window7_224_22k-aux5-768'
	valid = {
		0: [
			'/00013248.model.pth',
			'/00014766.model.pth',
			'/00012006.model.pth',
		],
		1: [
			'/00010488.model.pth',
			'/00011454.model.pth',
			'/00011868.model.pth',
		],
		2: [
			'/00014766.model.pth',
			'/00011454.model.pth',
			'/00013110.model.pth',
		],
		3: [
			'/00012006.model.pth',
			'/00014076.model.pth',
			'/00010488.model.pth',
		],
		4: [
			# '/00012006.model.pth',
			# '/00011454.model.pth',
			# '/00011454.model.pth',
		],
	}
	
	all_score_df = []
	for f, checkpoint in valid.items():
		if len(checkpoint)==0: continue
		
		project_name = out_dir.split('/')[-1]
		fold_dir = out_dir + '/fold-%d' % f
		
		checkpoint = [fold_dir + '/checkpoint/' + c for c in checkpoint]
		swa = do_swa(checkpoint)
		torch.save({
			'state_dict': swa,
			'swa': [c.split('/')[-1] for c in checkpoint],
		}, fold_dir + '/%s-fold-%d-swa.pth' % (project_name, f))
		
		## setup  ----------------------------------------
		submit_dir = fold_dir + '/valid/swa'
		os.makedirs(submit_dir, exist_ok=True)
		log = Logger()
		log.open(out_dir + '/log.submit.txt', mode='a')
		log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
		
		## dataset ------------------------------------
		train_df, valid_df = make_fold(f)
		
		## net ----------------------------------------
		log.write('** net setting **\n')
		if 1:
			net = Net().cuda()
			state_dict = swa
			net.load_state_dict(state_dict, strict=False)  # True
			
			net = net.eval()
			net.output_type = ['inference']
			
			# ----
			result = {
				'id': [],
				'probability': [],
				'rle': [],
			}
			
			start_timer = timer()
			for t, d in valid_df.iterrows():
				id = d['id']
				
				image = cv2.imread(root_dir + '/data/hubmap-organ-segmentation/train_images_png/%d.png' % id,
				                   cv2.IMREAD_COLOR)
				image = image.astype(np.float32) / 255
				H, W, _ = image.shape
				# image = cv2.cvtColor(read_tiff(tiff_file),cv2.COLOR_RGB2BGR)
				
				s = d.pixel_size / 0.4 * (image_size / 3000)
				# image = cv2.resize(image, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
				image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
				
				# we already probe that all test image are fixed size (within 0.8,1.2 of normalised size)
				# hence we just use resize. else use padding to 32 for variable size
				
				image = image_to_tensor(image)
				image = image.cuda()
				batch = {
					'image':
						torch.stack([
							image,
							torch.flip(image, [1]),
							torch.flip(image, [2]),
						]),  # simple TTA
				}
				
				probability = 0
				with torch.no_grad():
					with amp.autocast(enabled=is_amp):
						output = net(batch)  # data_parallel(net, batch) #
						# probability += output['probability']
						probability += F.interpolate(
							output['probability'], size=(H, W), mode='bilinear', align_corners=False, antialias=True)
				
				# undo TTA
				probability[1] = torch.flip(probability[1], [1])
				probability[2] = torch.flip(probability[2], [2])
				probability = probability.float().data.cpu().numpy().mean(0)[0]
				p = probability > organ_threshold[d.organ]
				rle = rle_encode(p)
				
				result['rle'].append(rle)
				result['probability'].append(probability)
				result['id'].append(id)
				print('\r', t, end='', flush=True)
			print('')
			# ---
			
			submit_df = pd.DataFrame({'id': result['id'], 'rle': result['rle']})
			print(submit_df)
			print('submit_df ok!')
			print('')
			
			submit_df.to_csv(submit_dir + '/submit_df.csv', index=False)
			valid_df.to_csv(submit_dir + '/valid_df.csv', index=False)
			probability = [p.astype(np.float16) for p in result['probability']]
			write_pickle_to_file(submit_dir + '/probability.pickle', probability)
		
		if 1:
			pass
			submit_df = pd.read_csv(submit_dir + '/submit_df.csv').fillna('')
			submit_df = submit_df.sort_values('id').reset_index(drop=True)
			truth_df = valid_df.sort_values('id').reset_index(drop=True)
			
			score_df = pd.DataFrame({'id': truth_df['id'], 'organ': truth_df['organ']})
			score_df.loc[:, 'fold'] = f
			score_df.loc[:, 'lb_score'] = 0
			
			num = len(submit_df)
			for i in range(num):
				t_df = truth_df.iloc[i]
				p_df = submit_df.iloc[i]
				t = rle_decode(t_df.rle, t_df.img_height, t_df.img_width, 1)
				p = rle_decode(p_df.rle, t_df.img_height, t_df.img_width, 1)
				
				dice = 2 * (t * p).sum() / (p.sum() + t.sum())
				score_df.loc[i, 'lb_score'] = dice
				
				if 0:
					image_show('t_mask', t_mask * 255, type='gray', resize=0.33)
					image_show('p_mask', p_mask * 255, type='gray', resize=0.33)
					cv2.waitKey(0)
			
			score_df.to_csv(submit_dir + '/score_df.csv', index=False)
			all_score_df.append(score_df)
			# print('lb_score', truth_df.lb_score.mean())
			
			log.write('organ_threshold = %s\n' % str(organ_threshold))
			log.write('swa checkpoint = %s\n' % str(checkpoint).replace(',', '\n'))
			log.write('lb_score = %s\n' % score_df.lb_score.mean())
			log.write('\n')
		
		if 0:  # learn best threshold
			probability = read_pickle_from_file(submit_dir + '/probability.pickle')
			
			threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
			more_df = pd.DataFrame({'id': valid_df['id'], 'organ': valid_df['organ']})
			for th in threshold:
				more_df.loc[:, 'dice_at%0.2f' % th] = 0
				
				num = len(valid_df)
				for i in range(num):
					print('\r', '%2d/%2d' % (i, num), end='', flush=True)
					t_df = valid_df.iloc[i]
					t = rle_decode(t_df.rle, t_df.img_height, t_df.img_width, 1)
					for th in threshold:
						p = probability[i] > th
						dice = 2 * (t * p).sum() / (p.sum() + t.sum())
						more_df.loc[i, 'dice_at%0.2f' % th] = dice
				
				print('')
				more_df.to_csv(submit_dir + '/more_df.csv', index=False)  # save intermediate
				# print('lb_score', more_df.lb_score.mean())
				zz = 0
			
			fig = plt.figure()
			ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # left, bottom, width, height (range 0 to 1)
			color = ['red', 'green', 'blue', 'orange', 'magenta', 'black']
			for i, organ in enumerate(['kidney', 'prostate', 'largeintestine', 'spleen', 'lung', 'all']):
				if organ != 'all':
					d = more_df[more_df.organ == organ]
				else:
					d = more_df
				v = d.mean(0).values[1:]
				ax.plot(threshold, v, color[i], label=organ)
				
				k = np.argmax(v)
				ax.scatter(threshold[k], v[k], s=30, c=color[i])
			
			# https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels-in-matplotlib
			ax.set_xticks(np.arange(0.10, 0.71, 0.10))
			ax.set_xticks(np.arange(0.10, 0.71, 0.02), minor=True)
			ax.set_yticks(np.arange(0.00, 1.01, 0.10))
			ax.set_yticks(np.arange(0.00, 1.01, 0.02), minor=True)
			ax.grid(which='minor', alpha=0.5)
			ax.grid(which='major', alpha=0.8)
			
			plt.ylim([0, 1.0])
			plt.legend()
			# plt.grid(linestyle = '--', linewidth = 0.5)
			
			fig.savefig(submit_dir + '/threshold-fig.png', )
			plt.waitforbuttonpress(1)
	
	# ---------------------
	all_score_df = pd.concat(all_score_df)
	print('oof')
	for organ in ['all', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
		if organ != 'all':
			d = all_score_df[all_score_df.organ == organ]
		else:
			d = all_score_df
		print('\t%f\t%s\t%f' % (len(d) / len(all_score_df), organ, d.lb_score.mean()))
	#print('')
	
	for f in [0,1,2,3,4]:
		print('fold-%d'%f)
		d0 = all_score_df[(all_score_df.fold == f)]
		for organ in ['all', 'kidney', 'prostate', 'largeintestine', 'spleen', 'lung']:
			if organ != 'all':
				d = d0[(d0.organ == organ)]
			else:
				d = d0
			print('\t%f\t%s\t%f' % (len(d) / len(all_score_df), organ, d.lb_score.mean()))
	#print('')
	
	zz = 0


# main #################################################################
if __name__ == '__main__':
	run_submit()
