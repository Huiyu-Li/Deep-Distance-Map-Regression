import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import csv
import numpy as np
import time
import shutil
import sys
import os
import math
from visdom import Visdom
from tqdm import tqdm
# my packages
from utils import *
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
    if config['saved_dir']  and os.path.exists(config['saved_dir']):
        shutil.rmtree(config['saved_dir'])

    logfile = os.path.join(ckpt_dir, 'log')
    sys.stdout = Logger(logfile)#see utils.py
    print(config)
    ###############GPU,Net,optimizer,scheduler###############
    seed = 0
    cudnn.benchmark = True
    # cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if torch.cuda.is_available():
        net = TumorNet().cuda()
        loss = MapDiceLoss().cuda()
    else:
        net = TumorNet()
        loss = DiceLoss()
    # net = DataParallel(net).cuda()
    # optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9,weight_decay=weight_decay)#SGD+Momentum
    optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30,
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
        test_loader = DataLoader(LiTSDataloader(dir_csv=config['valid_csv'],label_id=config['label_id']),
                                batch_size=config['batchSize_TVT'][2], shuffle=False, pin_memory=True)
        if config['saved_dir'] and not os.path.exists(config['saved_dir']):
            os.makedirs(config['saved_dir'])
        test_loss, test_dice, test_iter = test(test_loader, net, loss, config['saved_dir'])
        test_avgloss = sum(test_loss) / test_iter
        test_avgdice = sum(test_dice) / test_iter
        print("test_loss:%.4f, test_dice:%.4f, Time:%.3fmin " % (test_avgloss, test_avgdice, (time.time() - start_time) / 60))
        return

    # val_set_loader
    val_loader = DataLoader(LiTSDataloader(dir_csv=config['valid_csv'],label_id=config['label_id']),
                            batch_size=config['batchSize_TVT'][1], shuffle=False,pin_memory=True)
    #################train-eval (epoch)##############################
    max_validtumor= 0
    for epoch in range(config['max_epoches']):
        for epi in range(episode):
            # train_set_loader
            train_loader = DataLoader(LiTSDataloader(dir_csv=config['train_csv_list'][epi],label_id=config['label_id']),
                                    batch_size=config['batchSize_TVT'][0], shuffle=True, pin_memory=True)
            print('######train epoch-epi', str(epoch),'-',str(epi), 'lr=', str(optimizer.param_groups[0]['lr']),'######')
            train_loss, train_dice, train_iter = train(train_loader, net, loss, optimizer)
            train_avgloss = sum(train_loss) / train_iter
            train_avgdice = sum(train_dice) / train_iter
            print("[%d-%d/%d], train_loss:%.4f, train_dice:%.4f, Time:%.3fmin" %
                  (epoch, epi, config['max_epoches']-1, train_avgloss, train_avgdice, (time.time() - start_time) / 60))

            print('######valid epoch-epi', str(epoch),'-',str(epi),'######')
            valid_loss, valid_dice, valid_iter = validate(val_loader, net, loss)
            valid_avgloss = sum(valid_loss) / valid_iter
            valid_avgdice = sum(valid_dice) / valid_iter
            scheduler.step(valid_avgloss)
            print("[%d-%d/%d], valid_loss:%.4f, valid_dice:%.4f, Time:%.3fmin " %
                  (epoch, epi, config['max_epoches'], valid_avgloss, valid_avgdice, (time.time() - start_time) / 60))
            # print:lr,epoch/total,loss123,accurate,time

            if max_validtumor < abs(valid_avgdice):
                max_validtumor = abs(valid_avgdice)
                state = {
                    'epoche':epoch,
                    'arch':str(net),
                    'state_dict':net.state_dict(),
                    'optimizer':optimizer.state_dict()
                    #other measures
                }
                torch.save(state,ckpt_dir+'/checkpoint.pth.tar')
                #save model
                model_filename = ckpt_dir+'/model_'+str(epoch)+'-'+str(epi)+'-'+str(max_validtumor)[2:6]+'.pth'
                torch.save(net.state_dict(),model_filename)
                print('Model saved in',model_filename)
            viz.line([train_avgloss], [epoch * episode + epi], win='tloss',update='append')
            viz.line([train_avgdice], [epoch * episode + epi], win='tdice',update='append')
            viz.line([valid_avgloss], [epoch * episode + epi], win='vloss',update='append')
            viz.line([valid_avgdice], [epoch * episode + epi], win='vdice',update='append')

def train(data_loader, net, loss, optimizer):
    net.train()#swithch to train mode
    epoch_loss = []
    epoch_dice = []
    total_iter = len(data_loader)
    for i, (data, target, target_map,origin, direction, space, prefix, subNo) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            target_map = target_map.cuda()
        output = net(data)
        loss1 = loss(output, target_map)

        dice1 = Dice(output, target)
        optimizer.zero_grad()#set the grade to zero
        loss1.backward()
        optimizer.step()

        epoch_loss.append(loss1.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
        epoch_dice.append(dice1)

        num = len(prefix)
        name = [prefix[i].split('-')[-1] + '/' + subNo[i] for i in range(num)]
        print("[%d/%d], loss:%.4f, dice1:%.4f, name:%s" % (i, total_iter, loss1.item(),dice1,name))
    return epoch_loss,epoch_dice,total_iter

def validate(data_loader, net, loss):
    net.eval()
    epoch_loss = []
    epoch_dice = []
    total_iter = len(data_loader)
    with torch.no_grad():#no backward
        for i, (data, target, target_map,origin, direction, space, prefix, subNo) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_map = target_map.cuda()
            output = net(data)
            loss1 = loss(output, target_map)
            dice1 = Dice(output, target)

            epoch_loss.append(loss1.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
            epoch_dice.append(dice1)

            num = len(prefix)
            name = [prefix[i].split('-')[-1] + '/' + subNo[i] for i in range(num)]
            print("[%d/%d], loss:%.4f, dice1:%.4f, name:%s" % (i, total_iter,loss1.item(),dice1,name))
    return epoch_loss, epoch_dice, total_iter

def test(data_loader, net, loss, saved_dir):
    net.eval()
    epoch_loss = []
    epoch_dice = []
    total_iter = len(data_loader)
    with torch.no_grad():  # no backward
        for i, (data, target, target_map,origin, direction, space, prefix, subNo) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_map = target_map.cuda()
            output = net(data)
            loss1 = loss(output, target_map)
            dice1 = Dice(output, target)
            epoch_loss.append(loss1.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
            epoch_dice.append(dice1)

            num = len(prefix)
            name = [prefix[i].split('-')[-1] + '/' + subNo[i] for i in range(num)]
            print("[%d/%d], loss1:%.4f, dice1:%.4f, name:%s" % (i, total_iter,  loss1.item(), dice1, name))

            # check target, output1(0)+loss1, pred1+dice1
            pred1 = np.argmax(output.detach().cpu().numpy(), axis=1)#NDHW
            import matplotlib.pyplot as plt
            slice = 36
            plt.figure()
            plt.subplot(221);plt.imshow(target[0, 0,slice,:,:].detach().cpu().numpy());plt.title('target');plt.axis('off');plt.colorbar()
            plt.subplot(223);plt.imshow(output[0, 0, slice, :, :].detach().cpu().numpy());plt.title('output1-0'+str(loss1.item())[:4]);plt.axis('off');plt.colorbar()
            plt.subplot(224);plt.imshow(output[0, 1, slice, :, :].detach().cpu().numpy());plt.title('output1-1');plt.axis('off');plt.colorbar()
            plt.subplot(222);plt.imshow(pred1[0, slice, :, :]);plt.title('pred1:'+str(dice1)[:4]);plt.axis('off');plt.colorbar()
            plt.savefig(os.path.join(saved_dir,prefix[0]+'-'+subNo[0]+'.png'))
            plt.close()
            # save
            # for k in range(len(subNo)):
            #     saved_dir_new = os.path.join(saved_dir, prefix[0])
            #     if not os.path.exists(saved_dir_new):
            #         os.makedirs(saved_dir_new)
            #     output_name = os.path.join(saved_dir_new, subNo[k] + '.nii')
            #     saved_nii(output[k], origin[k], direction[k], space[k], output_name)
            #     print(output_name)
    return epoch_loss, epoch_dice, total_iter

if __name__ == '__main__':
    # print(torch.__version__)#0.4.1
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    viz = Visdom(env='G6 TumorNet-MapDice')
    viz.line([0], [0], win='tloss', opts = dict(title='train avgloss'))
    viz.line([0], [0], win='tdice', opts = dict(title='train avgdice'))
    viz.line([0], [0], win='vloss', opts = dict(title='valid avgloss'))
    viz.line([0], [0], win='vdice', opts = dict(title='valid avgdice'))
    ##########hyperparameters##########
    config_splitData = {
        'savedct_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/ct",
        'savedseg_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/seg",
        'savedmap_path': "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/NormInverse/",
        'TVTcsv' : './G2TVTcsv',
        'valid_csv' : './valid.csv',
        'episode' : 1,
        'ratio' : 0.9
    }
    ##########end hyperparameters##########
    # split_data_disMap(config_splitData)

    ##########hyperparameters##########
    episode = 1
    config = {
        'if_test' : False,
        'if_resume' : True,
        'max_epoches' : 300,
        'label_id': 2,
        'batchSize_TVT' : [2, 1, 1], #batchSize of train_valid_test
        'learning_rate' : 0.00001,#0.0001,
        'weight_decay' : 0,
        'train_csv_list': ['./G2TVTcsv/train' + str(i) + '.csv' for i in range(episode)],
        'valid_csv': './G2TVTcsv/valid.csv',
        'test_csv': './test.csv',
        # below is the saved path
        'ckpt_dir': './G00results/',
        'saved_dir': "/data/lihuiyu/LiTS/Valid_results/check822MapDiceN/",# will be rm before save
        'model_dir':"/home/lihuiyu/Code/STOAmodel/Stage1/TumorNetMaxpoolRelu-822.pth"
            #"/home/lihuiyu/Code/STOAmodel/Stage1/TumorNetMaxpoolRelu-822.pth"
            #"/home/lihuiyu/Code/STOAmodel/Stage1/20190924-083751-5999/model_0-9-5999.pth"
    }
    ##########hyperparameters##########
    from MyDataloader_Map import LiTSDataloader
    from TumorNet import *
    print(config['model_dir'])
    train_valid_seg(episode, config)#when parameters is tediously, just use config

    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))