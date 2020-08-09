import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
import csv
import numpy as np
import time
import shutil
import sys
import os
import math
from visdom import Visdom
# my packages
from utils import *

def one_hot(shape, labels):
    _labels = torch.zeros(shape)
    _labels.scatter_(dim=1, index=labels.long(), value=1)#scatter_(input, dim, index, src)
    return _labels
#################initialization network##############
def weights_init(model):
	if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
		nn.init.kaiming_uniform_(model.weight.data, 0.25)
		nn.init.constant_(model.bias.data, 0)
	# elif isinstance(model, nn.InstanceNorm3d):
	# 	nn.init.constant_(model.weight.data,1.0)
	# 	nn.init.constant_(model.bias.data, 0)

def train_valid_seg(episode,config):
	# refresh save dir
	exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
	ckpt_dir = os.path.join(config['ckpt_dir'] + exp_id)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if config['saved_dir'] and os.path.exists(config['saved_dir']):
		shutil.rmtree(config['saved_dir'])

	logfile = os.path.join(ckpt_dir, 'log')
	sys.stdout = Logger(logfile)#see utils.py
	###############GPU,Net,optimizer,scheduler###############
	torch.manual_seed(0)
	if torch.cuda.is_available():
		net = MapNet().cuda()#need to do this before constructing optimizer
		loss = Loss.cuda()
	else:
		net = MapNet()
		loss = Loss
	cudnn.benchmark = True  # True
	# net = DataParallel(net).cuda()
	# optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9,weight_decay=weight_decay)#SGD+Momentum
	optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#decay the learning rate after 100 epoches
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
														   verbose=False, threshold=0.0001, threshold_mode='rel',
														   cooldown=0, min_lr=0, eps=1e-08)
	###############resume or initialize prams###############
	if config['if_test'] or config['if_resume']:
		print('if_test:',config['if_test'],'if_resume:',config['if_resume'])
		checkpoint = torch.load(config['model_dir'])
		net.load_state_dict(checkpoint)
	else:
		print('weight initialization')
		net.apply(weights_init)

	#test
	if config['if_test']:
		print('###################test###################')
		test_loader = DataLoader(LiTSDataloader(dir_csv=config['valid_csv'],label_id = config['label_id']),
								batch_size=config['batchSize_TVT'][2], shuffle=False, pin_memory=True)
		if os.path.exists(config['saved_dir']):
			shutil.rmtree(config['saved_dir'])
		os.makedirs(config['saved_dir'])
		test_loss, test_tumor, test_iter = test(test_loader, net, loss, config['saved_dir'])
		test_avgloss = sum(test_loss) / test_iter
		test_avgtumor = sum(test_tumor) / test_iter
		print("test_loss:%.14f, test_tumor:%.4f, Time:%.3fmin" % (test_avgloss, test_avgtumor, (time.time() - start_time) / 60))
	return 

	# val_set_loader
	val_loader = DataLoader(LiTSDataloader(dir_csv=config['valid_csv']),
							batch_size=config['batchSize_TVT'][1], shuffle=False,pin_memory=True)
	#################train-eval (epoch)##############################
	# max_validtumor= 0.6
	min_loss = 1.
	max_iter = 0
	for epoch in range(config['max_epoches']):
		for epi in range(episode):
			# train_set_loader
			train_loader = DataLoader(LiTSDataloader(dir_csv=config['train_csv_list'][epi]),
									batch_size=config['batchSize_TVT'][0], shuffle=True, pin_memory=True)
			print('######train epoch-epi', str(epoch),'-',str(epi), 'lr=', str(optimizer.param_groups[0]['lr']),'######')
			train_loss, train_tumor, train_iter = train(train_loader, net, loss, optimizer)
			train_avgloss = sum(train_loss) / train_iter
			train_avgtumor = sum(train_tumor) / train_iter
			print("[%d-%d/%d], train_loss:%.14f, train_tumor:%.4f, Time:%.3fmin" %
				  (epoch, epi, config['max_epoches']-1, train_avgloss, train_avgtumor, (time.time() - start_time) / 60))

			print('######valid epoch-epi', str(epoch),'-',str(epi),'######')
			valid_loss, valid_tumor, valid_iter = validate(val_loader, net, loss, epoch, config['saved_dir'])
			valid_avgloss = sum(valid_loss) / valid_iter
			valid_avgtumor = sum(valid_tumor) / valid_iter
			scheduler.step(valid_avgloss)
			print("[%d-%d/%d], valid_loss:%.14f, valid_tumor:%.4f, Time:%.3fmin " %
				  (epoch, epi, config['max_epoches'], valid_avgloss, valid_avgtumor, (time.time() - start_time) / 60))
			# print:lr,epoch/total,loss123,accurate,time

			#if-save-model:
			# if epoch * episode + epi - max_iter == 10:
			# 	print("10 runs stagnation")
			# 	return
			if min_loss > abs(valid_avgloss):
				min_loss = abs(valid_avgloss)
				max_iter = epoch * episode + epi
				state = {
					'epoche':epoch,
					'arch':str(net),
					'state_dict':net.state_dict(),
					'optimizer':optimizer.state_dict()
					#other measures
				}
				torch.save(state,ckpt_dir+'/checkpoint.pth.tar')
				#save model
				model_filename = ckpt_dir+'/model_'+str(epoch)+'-'+str(epi)+'-'+str(min_loss)[2:6]+'.pth'
				torch.save(net.state_dict(),model_filename)
				print('Model saved in',model_filename)
			viz.line([train_avgloss], [epoch * episode + epi], win='train', update='append')
			viz.line([valid_avgloss], [epoch * episode + epi], win='valid', update='append')
			viz.line([train_avgtumor], [epoch * episode + epi], win='t_tumor', update='append')
			viz.line([valid_avgtumor], [epoch * episode + epi], win='v_tumor', update='append')

def train(data_loader, net, loss, optimizer):
	net.train()#swithch to train mode
	epoch_loss = []
	epoch_tumor = []
	total_iter = len(data_loader)
	for i, (target,target_map,origin,direction,space,prefix,subNo) in enumerate(data_loader):
		# extend input to two channel
		temp = 1. * (target >= 1)
		intend_shape = [target.shape[0],2,target.shape[2],target.shape[3],target.shape[4]]
		input = one_hot(intend_shape, temp)
		if torch.cuda.is_available():
			input = input.cuda()
			target = target.cuda()
			target_map = target_map.cuda()
		output = net(input)
		loss_output = loss(output, target_map)
		tumor_dice = MapDice(output, target)

		# # check input, target, target_map, output pred
		# pred = output.detach().cpu().numpy()
		# pred = np.where(pred > 0, 1., 0.).astype(np.float32)
		# import matplotlib.pyplot as plt
		# slice = 36
		# plt.figure()
		# # import pdb
		# # pdb.set_trace()
		# plt.subplot(231);plt.imshow(input[0, 0,slice,:,:].detach().cpu().numpy());plt.title('input-0');plt.axis('off');plt.colorbar()
		# plt.subplot(232);plt.imshow(input[0, 1, slice, :, :].detach().cpu().numpy());plt.title('input-1');plt.axis('off');plt.colorbar()
		# plt.subplot(233);plt.imshow(target[0,0,slice,:,:].detach().cpu().numpy());plt.title('target');plt.axis('off');plt.colorbar()
		# plt.subplot(234);plt.imshow(target_map[0, 0, slice, :, :].detach().cpu().numpy());plt.title('target_map');plt.axis('off');plt.colorbar()
		# plt.subplot(235);plt.imshow(output[0, 0, slice, :, :].detach().cpu().numpy());plt.title('output');plt.axis('off');plt.colorbar()
		# plt.subplot(236);plt.imshow(pred[0, 0, slice, :, :]);plt.title('pred');plt.axis('off');plt.colorbar()
		# plt.show()
		# plt.close()

		optimizer.zero_grad()#set the grade to zero
		loss_output.backward()
		optimizer.step()

		epoch_loss.append(loss_output.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
		epoch_tumor.append(tumor_dice)

		print("[%d/%d], loss:%.14f, tumor_dice:%.4f" % (i, total_iter, loss_output.item(), tumor_dice))

	return epoch_loss, epoch_tumor, total_iter

def validate(data_loader, net, loss, epoch, saved_dir):
    net.eval()
    epoch_loss = []
    epoch_tumor = []
    total_iter = len(data_loader)
    with torch.no_grad():#no backward
        for i, (target,target_map,origin,direction,space,prefix,subNo) in enumerate(data_loader):
            # extend input to two channel
            temp = 1. * (target >= 1)
            intend_shape = [target.shape[0], 2, target.shape[2], target.shape[3], target.shape[4]]
            input = one_hot(intend_shape, temp)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                target_map = target_map.cuda()
            output = net(input)
            loss_output = loss(output, target_map)
            tumor_dice = MapDice(output, target)

            epoch_loss.append(loss_output.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
            epoch_tumor.append(tumor_dice)

            print("[%d/%d], loss:%.14f, tumor_dice:%.4f" % (i, total_iter, loss_output.item(), tumor_dice))

            # if saved_dir and epoch % 10 == 0:
            #     for k in range(len(subNo)):
            #         saved_dir_new = os.path.join(saved_dir, prefix[k])
            #         if not os.path.exists(saved_dir_new):
            #             os.makedirs(saved_dir_new)
            #         output_name = os.path.join(saved_dir_new, subNo[k] + '.nii')
            #         saved_nii_map(output[k,:,:,:,:], origin[k,:], direction[k,:], space[k,:], output_name)
            #         print(output_name)

    return epoch_loss, epoch_tumor, total_iter

def test(data_loader, net, loss, saved_dir):
	net.eval()
	epoch_loss = []
	epoch_tumor = []
	total_iter = len(data_loader)
	with torch.no_grad():  # no backward
		for i, (target, target_map, origin, direction, space, prefix, subNo) in enumerate(data_loader):
			# extend input to two channel
			temp = 1. * (target >= 1)
			intend_shape = [target.shape[0], 2, target.shape[2], target.shape[3], target.shape[4]]
			input = one_hot(intend_shape, temp)
			if torch.cuda.is_available():
				input = input.cuda()
				target = target.cuda()
				target_map = target_map.cuda()
			output = net(input)
			loss_output = loss(output, target_map)
			tumor_dice = MapDice(output, target)
			epoch_loss.append(loss_output)
			epoch_tumor.append(tumor_dice)
			print("[%d/%d], loss:%.14f, tumor_dice:%.14f" % (i, total_iter, loss_output, tumor_dice))

			#check target target_map output
			import matplotlib.pyplot as plt
			plt.figure()
			slice = 36
			plt.subplot(221);plt.imshow(target[0,0,slice,:,:].detach().cpu().numpy());plt.title('target');plt.axis('off');plt.colorbar()
			plt.subplot(222);plt.imshow(target_map[0, 0, slice, :, :].detach().cpu().numpy());plt.title('target_map');plt.axis('off');plt.colorbar()
			plt.subplot(223);plt.imshow(output[0, 0, slice, :, :].detach().cpu().numpy());plt.title('output1');plt.axis('off');plt.colorbar()
			# plt.show()
			saved_dir_new = os.path.join(saved_dir, prefix[0])
			if not os.path.exists(saved_dir_new):
				os.makedirs(saved_dir_new)
			output_name = os.path.join(saved_dir_new, subNo[0] + '.png')
			plt.savefig(output_name)
			plt.close()
	return epoch_loss, epoch_tumor, total_iter
if __name__ == '__main__':
	# print(torch.__version__)#0.4.1
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
	start_time = time.time()

	os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	viz = Visdom(env='TumorNet G2 MapNet NormInverse')
	viz.line([0], [0], win='train',opts = dict(title='train avgloss'))
	viz.line([0], [0], win='valid', opts = dict(title='valid avgloss'))
	viz.line([0], [0], win='t_tumor', opts = dict(title='train avgtumor'))
	viz.line([0], [0], win='v_tumor', opts = dict(title='valid avgtumor'))
	##########hyperparameters##########
	config_splitData = {
		'savedseg_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/seg",
		'savedmap_path': "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/NormInverse/",
		'TVTcsv' : './GMTVTcsv',
		'valid_csv' : './valid.csv',
		'episode' : 1,
		'ratio' : 0.9
	}
	##########end hyperparameters##########
	print('######split_data:',config_splitData['savedseg_path'].split('/')[-2])
	split_data_MapNet(config_splitData)

	##########hyperparameters##########
	episode = 1
	config = {
		'if_test' : True,
		'if_resume' : True,
		'max_epoches' : 500,
		'label_id' :2,
		'batchSize_TVT' : [4, 4, 1], #batchSize of train_valid_test
		'CDHW' : [1, 48, 256, 256],
		'learning_rate' : 0.001,
		'weight_decay' : 0,
		'train_csv_list': ['./GMTVTcsv/train' + str(i) + '.csv' for i in range(episode)],
		'valid_csv': './GMTVTcsv/valid.csv',
		'test_csv': './test.csv',
		# below is the saved path
		'ckpt_dir': './GMresults/',
		'saved_dir': "/data/lihuiyu/LiTS/Valid_results/MapNet/",
		'model_dir': "/home/lihuiyu/Code/LiTS_TumorNet_Map/STOATumorMapNet/NormInverseMapNet/20200103-160314/model_93-0-3053.pth"
		}
	##########hyperparameters##########
	print(config['model_dir'])
	from MyDataloader_MapNet import LiTSDataloader
	from MapNet import *
	Loss = nn.SmoothL1Loss()
	train_valid_seg(episode, config)#when parameters is tediously, just use config

	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))