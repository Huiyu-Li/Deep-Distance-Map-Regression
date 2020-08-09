import csv
import math
import numpy as np
import os
import sys
import shutil

# split data
import re
def atoi(s):
	return int(s) if s.isdigit() else s

def natural_keys(text):
	return [atoi(c) for c in re.split('(\d+)', text)]

def split_data(config):
	#Split data into epi-train-valid-test
	ct_lists = os.listdir(config['savedct_path'])
	ct_lists.sort(key=natural_keys)
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for name in train_lists:
				# sub dir
				sub_list = os.listdir(os.path.join(config['savedct_path'], name))
				sub_list.sort(key=natural_keys)
				for sub in sub_list:
					ct_name = os.path.join(config['savedct_path'], name, sub)
					seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub)
					w.writerow((ct_name, seg_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for name in valid_lists:
			# sub dir
			sub_list = os.listdir(os.path.join(config['savedct_path'], name))
			sub_list.sort(key=natural_keys)
			for sub in sub_list:
				ct_name = os.path.join(config['savedct_path'], name, sub)
				seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub)
				w.writerow((ct_name, seg_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def split_data_disMap_fixedShuffle(config):
	#Split data into epi-train-valid-test
	ct_lists = os.listdir(config['savedct_path'])
	import random
	random.seed(9)
	random.shuffle(ct_lists)
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for name in train_lists:
				# sub dir
				sub_list = os.listdir(os.path.join(config['savedct_path'], name))
				sub_list.sort(key=natural_keys)
				sub_num = len(sub_list)
				for j in range(sub_num):
					ct_name = os.path.join(config['savedct_path'], name, sub_list[j])
					seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub_list[j])
					map_name = os.path.join(config['savedmap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii','npy'))
					w.writerow((ct_name, seg_name, map_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for name in valid_lists:
			# sub dir
			sub_list = os.listdir(os.path.join(config['savedct_path'], name))
			sub_list.sort(key=natural_keys)
			sub_num = len(sub_list)
			for j in range(sub_num):
				ct_name = os.path.join(config['savedct_path'], name, sub_list[j])
				seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub_list[j])
				map_name = os.path.join(config['savedmap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii','npy'))
				w.writerow((ct_name, seg_name, map_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def split_data_disMap(config):
	#Split data into epi-train-valid-test
	ct_lists = os.listdir(config['savedct_path'])
	ct_lists.sort(key=natural_keys)
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for name in train_lists:
				# sub dir
				sub_list = os.listdir(os.path.join(config['savedct_path'], name))
				sub_list.sort(key=natural_keys)
				sub_num = len(sub_list)
				for j in range(sub_num):
					ct_name = os.path.join(config['savedct_path'], name, sub_list[j])
					seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub_list[j])
					map_name = os.path.join(config['savedmap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii','npy'))
					w.writerow((ct_name, seg_name, map_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for name in valid_lists:
			# sub dir
			sub_list = os.listdir(os.path.join(config['savedct_path'], name))
			sub_list.sort(key=natural_keys)
			sub_num = len(sub_list)
			for j in range(sub_num):
				ct_name = os.path.join(config['savedct_path'], name, sub_list[j])
				seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub_list[j])
				map_name = os.path.join(config['savedmap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii','npy'))
				w.writerow((ct_name, seg_name, map_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def split_data_BidisMap(config):
	#Split data into epi-train-valid-test
	ct_lists = os.listdir(config['savedct_path'])
	ct_lists.sort(key=natural_keys)
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for name in train_lists:
				# sub dir
				sub_list = os.listdir(os.path.join(config['savedct_path'], name))
				sub_list.sort(key=natural_keys)
				sub_num = len(sub_list)
				for j in range(sub_num):
					ct_name = os.path.join(config['savedct_path'], name, sub_list[j])
					seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub_list[j])
					bimap_name = os.path.join(config['savedbimap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii', 'npy'))
					map_name = os.path.join(config['savedmap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii','npy'))
					w.writerow((ct_name, seg_name, bimap_name, map_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for name in valid_lists:
			# sub dir
			sub_list = os.listdir(os.path.join(config['savedct_path'], name))
			sub_list.sort(key=natural_keys)
			sub_num = len(sub_list)
			for j in range(sub_num):
				ct_name = os.path.join(config['savedct_path'], name, sub_list[j])
				seg_name = os.path.join(config['savedseg_path'], name.replace('volume', 'segmentation'), sub_list[j])
				bimap_name = os.path.join(config['savedbimap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii', 'npy'))
				map_name = os.path.join(config['savedmap_path'], name.replace('volume', 'segmentation'), sub_list[j].replace('nii','npy'))
				w.writerow((ct_name, seg_name, bimap_name, map_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def split_data_MapNet(config):
	#Split data into epi-train-valid-test
	ct_lists = os.listdir(config['savedseg_path'])
	ct_lists.sort(key=natural_keys)
	total = len(ct_lists)

	tn = math.ceil(total * config['ratio'])
	tn_epi = tn // config['episode']
	tn = tn_epi * config['episode']  # remove the train tail
	valid_lists = ct_lists[tn:total]

	# clear the exists file
	if os.path.isdir(config['TVTcsv']):
		shutil.rmtree(config['TVTcsv'])
	os.mkdir(config['TVTcsv'])
	train_csv_list = ['train' + str(i) + '.csv' for i in range(config['episode'])]
	for epi in range(config['episode']):
		train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]  # attention:[0:num_train)
		with open(os.path.join(config['TVTcsv'], train_csv_list[epi]), 'w') as file:
			w = csv.writer(file)
			for name in train_lists:
				# sub dir
				sub_list = os.listdir(os.path.join(config['savedseg_path'], name))
				sub_list.sort(key=natural_keys)
				sub_num = len(sub_list)
				for j in range(sub_num):
					seg_name = os.path.join(config['savedseg_path'], name, sub_list[j])
					map_name = os.path.join(config['savedmap_path'], name,sub_list[j].replace('nii','npy'))
					w.writerow((seg_name, map_name))  # attention: the first row defult to tile
	with open(os.path.join(config['TVTcsv'], config['valid_csv']), 'w') as file:
		w = csv.writer(file)
		for name in valid_lists:
			# sub dir
			sub_list = os.listdir(os.path.join(config['savedseg_path'], name))
			sub_list.sort(key=natural_keys)
			sub_num = len(sub_list)
			for j in range(sub_num):
				seg_name = os.path.join(config['savedseg_path'], name, sub_list[j])
				map_name = os.path.join(config['savedmap_path'], name,sub_list[j].replace('nii','npy'))
				w.writerow((seg_name, map_name))  # attention: the first row defult to tile
	print('total=', total, 'train=', tn, '(', tn_epi, '*', config['episode'], ')', 'val=', len(valid_lists))

def getFreeId():
    import pynvml

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput == 'all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu ' + 'nx-x' + 'is being used') #//ValueError('gpu ' + g + 'is being used')
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message) #print to screen
        self.log.write(message) #print to logfile

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass