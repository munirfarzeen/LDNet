import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dropblock import DropBlock2D, LinearScheduler
import torchvision.models as models



def init_weights(net, init_type='normal', gain=0.02):
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)

class conv_block(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(conv_block,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			# nn.Dropout2d(0.2)
		)


	def forward(self,x):
		x = self.conv(x)
		return x

class conv_block_dia(nn.Module):
	def __init__(self,ch_in,ch_out,rate):
		super(conv_block_dia,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=rate,dilation=rate,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True),
			nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=rate,dilation=rate,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
			# nn.Dropout2d(0.2)
		)


	def forward(self,x):
		x = self.conv(x)
		return x

class ASPPConv(nn.Sequential):
	def __init__(self, in_channels, out_channels, dilation):
		modules = [
			nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
			# nn.Dropout2d(0.2)
			
		]
		super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
	def __init__(self, in_channels, out_channels):
		super(ASPPPooling, self).__init__(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU())

	def forward(self, x):
		size = x.shape[-2:]
		for mod in self:
			x = mod(x)
		return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class up_conv(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(up_conv,self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self,x):
		x = self.up(x)
		return x

class up_conv_vgg(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(up_conv_vgg,self).__init__()
		self.up = nn.Sequential(
			nn.Upsample(scale_factor=32),
			nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self,x):
		x = self.up(x)
		return x

class single_conv(nn.Module):
	def __init__(self,ch_in,ch_out):
		super(single_conv,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
			nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
		)

	def forward(self,x):
		x = self.conv(x)
		return x

class Attention_block(nn.Module):
	def __init__(self,F_g,F_l,F_int):
		super(Attention_block,self).__init__()
		self.W_g = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
			)
		
		self.W_x = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(F_int)
		)

		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self,g,x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.relu(g1+x1)
		psi = self.psi(psi)

		return x*psi



class LDNet(nn.Module):
	def __init__(self,n_classes=5,img_ch=1,output_ch=1):
		super(LDNet,self).__init__()
		
		self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
		# self.features=models.vgg16(pretrained=True).features
		# self.features[0]=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
		# self.upsample=up_conv_vgg(ch_in=512,ch_out=32)

		self.Conv1 = conv_block(ch_in=3,ch_out=32)
		self.Conv2 = conv_block_dia(ch_in=32,ch_out=64,rate=2)
		self.Conv3 = conv_block_dia(ch_in=64,ch_out=128,rate=4)
		self.Conv4 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=True)

		self.box1=ASPPConv(in_channels=256,out_channels=256,dilation=1)
		self.box2=ASPPConv(in_channels=256,out_channels=256,dilation=2)
		self.box3=ASPPConv(in_channels=256,out_channels=256,dilation=4)
		self.box4=ASPPConv(in_channels=256,out_channels=256,dilation=8)
		self.box5=ASPPConv(in_channels=256,out_channels=256,dilation=16)
		self.box6=ASPPConv(in_channels=256,out_channels=256,dilation=32)


		# self.Conv4 = conv_block(ch_in=256,ch_out=512)
		# self.Conv5 = conv_block(ch_in=512,ch_out=1024)

		# self.Up5 = up_conv(ch_in=1024,ch_out=512)
		# self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

		self.Conv5 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=True)
		# self.Conv5 = conv_block(ch_in=512,ch_out=1024)

		# self.Up5 = up_conv(ch_in=1024,ch_out=512)
		# self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
		# self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

		self.Up4 = up_conv(ch_in=256,ch_out=128)
		self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
		self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
		
		self.Up3 = up_conv(ch_in=128,ch_out=64)
		self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
		self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
		
		self.Up2 = up_conv(ch_in=64,ch_out=32)
		self.Att2 = Attention_block(F_g=32,F_l=32,F_int=1)
		self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

		self.Conv_1x1 = nn.Conv2d(32,n_classes,kernel_size=1,stride=1,padding=0)
		self.dropblock = LinearScheduler(
            DropBlock2D( block_size=5,drop_prob=0.),
            start_value=0.,
            stop_value=0.5,
            nr_steps=5000
        )


	def forward(self,x):
		self.dropblock.step()
		
		# encoding path
		# xf=self.features(x)
		# xs=self.upsample(xf)
		x1 = self.Conv1(x)
		x1=self.dropblock (x1)
		x2 = self.Maxpool(x1)

		x2 = self.Conv2(x2)
		x2=self.dropblock (x2)
		x3 = self.Maxpool(x2)

		x3 = self.Conv3(x3)
		x3=self.dropblock (x3)
		x4 = self.Maxpool(x3)

		x4 = self.Conv4(x4)
		x4=self.dropblock (x4)
		
		b1=self.box1(x4)
		b2=self.box2(x4)
		b3=self.box3(x4)
		b4=self.box4(x4)
		b5=self.box5(x4)
		b6=self.box6(x4)
		xb=b1+b2+b3+b4+b5+b6
		xb=self.dropblock (xb)

		d5=self.Conv5(xb)
		d4 = self.Up4(d5)
		x3 = self.Att4(g=d4,x=x3)
		d4 = torch.cat((x3,d4),dim=1)

		d4 = self.Up_conv4(d4)
		d3 = self.Up3(d4)
		x2 = self.Att3(g=d3,x=x2)
		d3 = torch.cat((x2,d3),dim=1)

		d3 = self.Up_conv3(d3)
		d2 = self.Up2(d3)
		x1 = self.Att2(g=d2,x=x1)
		d2 = torch.cat((x1,d2),dim=1)

		d2 = self.Up_conv2(d2)
		d1 = self.Conv_1x1(d2)
		# d11 = F.interpolate(d1, size=x.size()[2:], mode='bilinear', align_corners=True)

		return d1

