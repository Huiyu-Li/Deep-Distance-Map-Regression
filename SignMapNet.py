# # MapNet2 two channel input, one channel output
# two input channel to simulate the output of softmax [bg,fg]
import torch
from torch import nn
import numpy as np
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import distance_transform_edt
def saved_nii(savedImg,origin,direction,xyz_thickness,saved_name):
	origin = tuple(k.item() for k in origin)
	direction = tuple(k.item() for k in direction)
	xyz_thickness = tuple(k.item() for k in xyz_thickness)
	savedImg = np.argmax(savedImg.detach().cpu().numpy(),axis=0).astype(np.float32)
	newImg = sitk.GetImageFromArray(savedImg)
	newImg.SetOrigin(origin)
	newImg.SetDirection(direction)
	newImg.SetSpacing(xyz_thickness)
	sitk.WriteImage(newImg, saved_name)

def saved_nii_map(savedImg,origin,direction,xyz_thickness,saved_name,threshold=0):
    origin = tuple(k.item() for k in origin)
    direction = tuple(k.item() for k in direction)
    xyz_thickness = tuple(k.item() for k in xyz_thickness)
    savedImg = savedImg.detach().cpu().numpy()
    savedImg = np.where(savedImg > threshold, 1., 0.).astype(np.float32)
    newImg = sitk.GetImageFromArray(savedImg)
    newImg.SetOrigin(origin)
    newImg.SetDirection(direction)
    newImg.SetSpacing(xyz_thickness)
    sitk.WriteImage(newImg, saved_name)

def MapDice(output, target, threshold=0):
    pred = output.detach().cpu().numpy()
    pred = np.where(pred > threshold, 1., 0.)
    target = target.detach().cpu().numpy()
    # Compute per-case (per patient volume) dice.
    if pred.shape != target.shape:
        raise AttributeError("Shapes do not match!")
    if not np.any(pred) and not np.any(target):
        dice = 1.
        print('dice = 1')
    else:
        dice = metric.dc(pred, target)
    return dice

class ContourMapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, dismap, contour):
        loss = -((contour * dismap).sum() + self.smooth) / (contour.sum() + self.smooth) # 归一化
        return loss

class CosineMapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5

    def forward(self, dismap, target_map):
        loss = -((dismap * target_map).sum() + self.smooth) / (torch.sqrt((dismap*dismap).sum())*torch.sqrt((target_map*target_map).sum()) + self.smooth) # 归一化
        print('loss',loss.item())#'log',torch.log(loss).item()=nan
        return loss

class MapDiceLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.smooth = 1e-5

	def forward(self, output, target):
		# splited channel0 is bg
		intersection_fg = 2. * (output * target).sum()
		denominator_fg = output.sum() + target.sum()
		dice = (intersection_fg + self.smooth) / (denominator_fg + self.smooth)
		dice = 1 - dice
		return dice

class MapSalienceLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.smooth = 1e-5

	def forward(self, output, target, map):
		# splited channel0 is bg
		intersection_fg = 2. * (output * target)
		denominator_fg = output + target
		dice = (map*(intersection_fg + self.smooth) / (denominator_fg + self.smooth)).sum()
		dice = 1 - dice
		return dice

class PostRes(nn.Module):
	def __init__(self, n_in, n_out, stride = 1):
		super(PostRes, self).__init__()
		self.resBlock = nn.Sequential(
			nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1),
			nn.InstanceNorm3d(n_out),
			nn.ReLU(inplace=True),
			# nn.PReLU(),
			nn.Conv3d(n_out, n_out, kernel_size=3, padding=1),
			nn.InstanceNorm3d(n_out)
		)
		self.relu = nn.ReLU(inplace=True)
		# self.prelu = nn.PReLU()

		if stride != 1 or n_out != n_in:
			self.shortcut = nn.Sequential(
				nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
				nn.InstanceNorm3d(n_out))
		else:
			self.shortcut = None

	def forward(self, x):
		residual = x
		if self.shortcut is not None:
			residual = self.shortcut(x)

		out = self.resBlock(x)
		out += residual
		out = self.relu(out)
		# out = self.prelu(out)
		return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_blocks_back = [2]
        self.nff = [2, 4, 8]# NumFeature_Forw
        self.outChannel = 1
        #deconv,output
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(self.nff[-1], self.nff[-2], kernel_size=2, stride=2),
            nn.InstanceNorm3d(self.nff[-2]),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv3d(self.nff[-1], self.nff[-2], kernel_size=1),
            nn.InstanceNorm3d(self.nff[-2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.nff[-2], self.outChannel, kernel_size=1),
            # nn.ReLU(inplace=True)
            nn.Tanh()
        )
        #backward
        num_back = len(self.num_blocks_back)
        for i in range(num_back):
            blocks = []
            for j in range(self.num_blocks_back[i]):
                blocks.append(PostRes(self.nff[-(i+1)], self.nff[-(i+1)]))
            setattr(self, 'backward' + str(num_back - i), nn.Sequential(*blocks))

    def forward(self, layer1, layer2):
        # # decoder
        # up2 = self.deconv2(layer_3)
        # cat_2 = torch.cat((up2, layer2), 1)
        # layer_2 = self.backward2(cat_2)
        up1 = self.deconv1(layer2)
        cat_1 = torch.cat((up1, layer1), 1)
        layer_1 = self.backward1(cat_1)

        layer_1 = self.output(layer_1)
        return layer_1

class MapNet(nn.Module):
    def __init__(self):
        super(MapNet, self).__init__()
        self.nff = [2, 4, 8]#NumFeature_Forw
        self.num_blocks_forw = [2]
        # forward1
        self.forward1 = nn.Sequential(
            nn.Conv3d(self.nff[0], self.nff[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.nff[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.nff[1], self.nff[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.nff[1]),
            nn.ReLU(inplace=True)
        )
        # forward2-5
        for i in range(len(self.num_blocks_forw)):
            blocks = []
            for j in range(self.num_blocks_forw[i]):
                blocks.append(PostRes(self.nff[i + 2], self.nff[i + 2]))
            setattr(self, 'forward' + str(i + 2), nn.Sequential(*blocks))

        # downsamp by stride convolution
        self.downsamp1 = nn.Conv3d(self.nff[1], self.nff[2], kernel_size=3, stride=2, padding=1)
        #self.downsamp2 = nn.Conv3d(self.nff[2], self.nff[3], kernel_size=3, stride=2, padding=1)

        self.decoder = Decoder()

    def forward(self, input):
        #encoder
        layer1 = self.forward1(input)
        down1 = self.downsamp1(layer1)

        layer2 = self.forward2(down1)
        #down2 = self.downsamp2(layer2)

        #layer3 = self.forward3(down2)

        # decoder
        branch2 = self.decoder(layer1, layer2)
        return branch2

def main():
    net = MapNet().cuda()#necessary for torchsummary, must to cuda

    # from torchsummary import summary
    # summary(net, input_size=(2,48,256,256))#must remove the number of N

    # input = torch.randn([7,2,48,256,256]).cuda()#(NCDHW)
    # output = net(input)
    # print(output.shape)

    for name, param in net.named_parameters():
    	print(name)


if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	main()