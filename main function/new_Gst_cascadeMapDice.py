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
    print(config)
    ###############GPU,Net,optimizer,scheduler###############
    torch.manual_seed(0)
    if torch.cuda.is_available():
        net = TumorNet().cuda()#need to do this before constructing optimizer
        net_tail = MapNet().cuda()
        mapDiceLoss = MapDiceLoss().cuda()
    else:
        net = TumorNet()
        net_tail = MapNet()
    cudnn.benchmark = True  # True
    # net = DataParallel(net).cuda()
    # 记得检查一下是否冻结成功。
    for param in net_tail.parameters():
        param.requires_grad = False
    # filter
    optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'], betas=(0.9, 0.999), eps=1e-08,weight_decay=config['weight_decay'])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#decay the learning rate after 100 epoches
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8,patience=30,verbose=False, threshold=0.0001,
                                                           threshold_mode='rel',cooldown=0, min_lr=0, eps=1e-08)
    ###############resume or initialize prams###############
    if config['if_test'] or config['if_resume']:
        print('if_test:',config['if_test'],'if_resume:',config['if_resume'])
        checkpoint = torch.load(config['model_dir'])
        print(config['model_dir'])
        net.load_state_dict(checkpoint)
        checkpoint2 = torch.load(config['tail_dir'])
        print(config['tail_dir'])
        net_tail.load_state_dict(checkpoint2)
    else:
        print('weight initialization')
        net.apply(weights_init)

    #test
    if config['if_test']:
        print('###################test###################')
        test_loader = DataLoader(LiTSDataloader(dir_csv=config['valid_csv'],label_id=config['label_id']),
                                batch_size=config['batchSize_TVT'][2], shuffle=False, pin_memory=True)
        if config['saved_dir']:
            if os.path.exists(config['saved_dir']):
                shutil.rmtree(config['saved_dir'])
            os.makedirs(config['saved_dir'])
        test_mapDice, test_dice1, test_dice2, test_iter  = test(test_loader, net, net_tail, mapDiceLoss, config['saved_dir'])
        test_avgmapDice = sum(test_mapDice) / test_iter
        test_avgdice1 = sum(test_dice1) / test_iter
        test_avgdice2 = sum(test_dice2) / test_iter
        print("test_mapDice:%.4f, test_dice1:%.4f, test_dice2:%.4f, Time:%.3fmin" %
              (test_avgmapDice, test_avgdice1, test_avgdice2, (time.time() - start_time) / 60))
        return

    # val_set_loader
    val_loader = DataLoader(LiTSDataloader(dir_csv=config['valid_csv'],label_id=config['label_id']),
                            batch_size=config['batchSize_TVT'][1], shuffle=False,pin_memory=True)
    #################train-eval (epoch)##############################
    min_loss= 2
    for epoch in range(config['max_epoches']):
        for epi in range(episode):
            # train_set_loader
            train_loader = DataLoader(LiTSDataloader(dir_csv=config['train_csv_list'][epi],label_id=config['label_id']),
                                    batch_size=config['batchSize_TVT'][0], shuffle=True, pin_memory=True)
            print('######train epoch-epi', str(epoch),'-',str(epi), 'lr=', str(optimizer.param_groups[0]['lr']),'######')
            train_mapDice, train_dice1, train_dice2, train_iter = train(train_loader, net, net_tail, mapDiceLoss, optimizer)
            train_avgmapDice = sum(train_mapDice) / train_iter
            train_avgdice1 = sum(train_dice1) / train_iter
            train_avgdice2 = sum(train_dice2) / train_iter
            print("[%d-%d/%d], train_mapDice:%.4f, train_dice1:%.4f, train_dice2:%.4f, Time:%.3fmin" %
                  (epoch, epi, config['max_epoches']-1, train_avgmapDice, train_avgdice1, train_avgdice2,(time.time() - start_time) / 60))

            print('######valid epoch-epi', str(epoch),'-',str(epi),'######')
            valid_mapDice, valid_dice1, valid_dice2, valid_iter = validate(val_loader, net, net_tail, mapDiceLoss)
            valid_avgmapDice = sum(valid_mapDice) / valid_iter
            valid_avgdice1 = sum(valid_dice1) / valid_iter
            valid_avgdice2 = sum(valid_dice2) / valid_iter
            scheduler.step(valid_avgmapDice)
            print("[%d-%d/%d], valid_mapDice:%.4f, valid_dice1:%.4f, valid_dice2:%.4f,Time:%.3fmin " %
                  (epoch, epi, config['max_epoches'], valid_avgmapDice, valid_avgdice1, valid_avgdice2,(time.time() - start_time) / 60))

            #if-save-model:
            if train_avgmapDice < min_loss:
                min_loss = train_avgmapDice
                # max_iter = epoch * episode + epi
                state = {
                    'epoche':epoch,
                    'arch':str(net),
                    'state_dict':net.state_dict(),
                    'optimizer':optimizer.state_dict()
                    #other measures
                }
                torch.save(state,ckpt_dir+'/checkpoint.pth.tar')
                #save model
                model_filename = ckpt_dir+'/model_'+str(epoch)+'-'+str(epi)+'-'+str(abs(min_loss))[2:6]+'.pth'
                torch.save(net.state_dict(),model_filename)
                print('Model saved in',model_filename)
                # save model
                model_filename = ckpt_dir + '/tail_' + str(epoch) + '-' + str(epi) + '-' + str(valid_avgdice1)[2:6] + '.pth'
                torch.save(net_tail.state_dict(), model_filename)
                print('Model saved in', model_filename)

            viz.line([train_avgmapDice], [epoch * episode + epi], win='tloss', name= 'mapDice',update='append')
            viz.line([train_avgdice1], [epoch * episode + epi], win='tdice',name='dice1',update='append')
            viz.line([train_avgdice2], [epoch * episode + epi], win='tdice',name='dice2',update='append')
            viz.line([valid_avgmapDice], [epoch * episode + epi], win='vloss', name= 'mapDice',update='append')
            viz.line([valid_avgdice1], [epoch * episode + epi], win='vdice',name='dice1',update='append')
            viz.line([valid_avgdice2], [epoch * episode + epi], win='vdice',name='dice2',update='append')

def train(data_loader, net, net_tail, mapDiceLoss, optimizer):
    net.train()#swithch to train mode
    epoch_mapDice = []
    epoch_dice1 = []
    epoch_dice2 = []
    total_iter = len(data_loader)
    for i, (data,target,target_map,origin,direction,space,prefix,subNo) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            target_map = target_map.cuda()
        output1 = net(data)
        dice1 = Dice(output1, target)
        epoch_dice1.append(dice1)

        # tail
        output2 = net_tail(output1)
        loss_mapDice = mapDiceLoss(output2, target_map)
        dice2 = MapDice(output2, target)

        epoch_mapDice.append(loss_mapDice.item())
        epoch_dice2.append(dice2)

        optimizer.zero_grad()  # set the grade to zero
        loss_mapDice.backward()
        optimizer.step()

        num = len(prefix)
        name = [prefix[i].split('-')[-1] + '/' + subNo[i] for i in range(num)]
        print("[%d/%d], loss_mapDice:%.4f, dice1:%.4f, dice2:%.4f, name:%s" %
              (i, total_iter, loss_mapDice.item(), dice1, dice2, name))
    return epoch_mapDice, epoch_dice1, epoch_dice2, total_iter

def validate(data_loader, net, net_tail, mapDiceLoss):
    net.eval()
    epoch_mapDice = []
    epoch_dice1 = []
    epoch_dice2 = []
    total_iter = len(data_loader)
    with torch.no_grad():#no backward
        for i, (data,target,target_map,origin,direction,space,prefix,subNo) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_map = target_map.cuda()
            output1 = net(data)
            dice1 = Dice(output1, target)
            epoch_dice1.append(dice1)

            # tail
            output2 = net_tail(output1)
            loss_mapDice = mapDiceLoss(output2, target_map)
            dice2 = MapDice(output2, target)

            epoch_mapDice.append(loss_mapDice.item())
            epoch_dice2.append(dice2)

            num = len(prefix)
            name = [prefix[i].split('-')[-1] + '/' + subNo[i] for i in range(num)]
            print("[%d/%d], loss_mapDice:%.4f, dice1:%.4f, dice2:%.4f, name:%s" %
                (i, total_iter, loss_mapDice.item(), dice1, dice2, name))
    return epoch_mapDice, epoch_dice1, epoch_dice2, total_iter

def test(data_loader, net, net_tail, mapDiceLoss, saved_dir):
    net.eval()
    epoch_mapDice = []
    epoch_dice1 = []
    epoch_dice2 = []
    total_iter = len(data_loader)
    with torch.no_grad():  # no backward
        for i, (data, target, target_map,origin, direction, space, prefix, subNo) in enumerate(data_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                target_map = target_map.cuda()
            output1 = net(data)
            dice1 = Dice(output1, target)
            epoch_dice1.append(dice1)

            # tail
            output2 = net_tail(output1)
            loss_mapDice = mapDiceLoss(output2, target_map)
            dice2 = MapDice(output2, target)

            epoch_mapDice.append(loss_mapDice.item())
            epoch_dice2.append(dice2)

            num = len(prefix)
            name = [prefix[i].split('-')[-1] + '/' + subNo[i] for i in range(num)]
            print(
                "[%d/%d], loss_mapDice:%.4f, dice1:%.4f, dice2:%.4f, name:%s" %
                (i, total_iter, loss_mapDice.item(), dice1, dice2, name))

            # check target, target_map,
            # output1(0)+loss1, pred1+dice1, output2+loss2, pred2+dice2
            pred1 = np.argmax(output1.detach().cpu().numpy(), axis=1)#NDHW
            pred2 = output2.detach().cpu().numpy()
            pred2 = np.where(pred2 > 0, 1., 0.).astype(np.float32)
            import matplotlib.pyplot as plt
            slice = 36
            plt.figure()
            plt.subplot(321);plt.imshow(target[0,0,slice,:,:].detach().cpu().numpy());plt.title('target');plt.axis('off');plt.colorbar()
            plt.subplot(322);plt.imshow(target_map[0, 0, slice, :, :].detach().cpu().numpy());plt.title('target_map');plt.axis('off');plt.colorbar()
            plt.subplot(323);plt.imshow(output1[0, 0, slice, :, :].detach().cpu().numpy());plt.title('output1-0');plt.axis('off');plt.colorbar()
            # plt.subplot(424);plt.imshow(output1[0, 1, slice, :, :].detach().cpu().numpy());plt.title('output1-1');plt.axis('off');plt.colorbar()
            plt.subplot(324);plt.imshow(pred1[0, slice, :, :]);plt.title('pred1:'+str(dice1)[:4]);plt.axis('off');plt.colorbar()
            plt.subplot(325);plt.imshow(output2[0, 0, slice, :, :].detach().cpu().numpy());plt.title('output2:'+str(loss_mapDice.item())[:4]);plt.axis('off');plt.colorbar()
            plt.subplot(326);plt.imshow(pred2[0, 0, slice, :, :]);plt.title('pred2:'+str(dice2)[:4]);plt.axis('off');plt.colorbar()
            plt.savefig(os.path.join(saved_dir,prefix[0]+'-'+subNo[0]+'.png'))
            plt.close()
            # # save
    return epoch_mapDice, epoch_dice1, epoch_dice2, total_iter

if __name__ == '__main__':
	# print(torch.__version__)#0.4.1
	start_time = time.time()
	os.environ["CUDA_VISIBLE_DEVICES"] = "2"
	viz = Visdom(env='G2 TumorNet cascade frozen-MapNet-MapDice')
	viz.line([0], [0], win='tloss',opts=dict(title='train avgloss', legend=['mapDice']))
	viz.line([0], [0], win='tdice', opts=dict(title='train avgdice', legend=['dice1', 'dice2']))
	viz.line([0], [0], win='vloss',opts=dict(title='valid avgloss', legend=['mapDice']))
	viz.line([0], [0], win='vdice', opts=dict(title='valid avgdice', legend=['dice1', 'dice2']))
	##########hyperparameters##########
	config_splitData = {
		'savedct_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/ct",
		'savedseg_path' : "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/seg",
		'savedmap_path': "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/SignNormInverse/",
		'TVTcsv' : './G3TVTcsv',
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
		'max_epoches' : 100,
        'label_id': 2,
		'batchSize_TVT' : [2, 1, 1], #batchSize of train_valid_test
		'learning_rate' : 0.00001,
		'weight_decay' : 0,
		'train_csv_list': ['./G3TVTcsv/train' + str(i) + '.csv' for i in range(episode)],
		'valid_csv': './G3TVTcsv/valid.csv',
		'test_csv': './test.csv',
		# below is the saved path
		'ckpt_dir': './G00results/',
		'saved_dir': "/data/lihuiyu/LiTS/Valid_results/check822mapDiceN/",#"/data/lihuiyu/LiTS/Valid_results/checkDiceCosine0/",
        'model_dir':"/home/lihuiyu/Code/STOAmodel/Stage1/TumorNetMaxpoolRelu-822.pth",
            # "/home/lihuiyu/Code/STOAmodel/Stage1/20190924-083751-5999/model_0-9-5999.pth",
        'tail_dir':"/home/lihuiyu/Code/LiTS_TumorNet_Map/STOATumorMapNet/SignNormInverse/20200109-193132/model_14-0-0159.pth"
        # 'model_dir': ,
        # 'tail_dir':
	}
	##########hyperparameters##########
	from MyDataloader_Map import LiTSDataloader
	from TumorNet import TumorNet, Dice, DiceLoss
	from SignMapNet import *
	train_valid_seg(episode, config)

	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))