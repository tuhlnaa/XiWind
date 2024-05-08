"""
High Quality Monocular Depth Estimation via Transfer Learning
https://arxiv.org/abs/1812.11941
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from torchvision.models import densenet161, DenseNet161_Weights

class Encoder(nn.Module):
	def __init__(self, encoder_pretrained=True):
		super(Encoder, self).__init__()
		self.densenet = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)

	def forward(self, x):
		feature_maps = [x]
		# for key, value in self.densenet.features._modules.items():
		# 	feature_maps.append(value(feature_maps[-1]))
		for module in self.densenet.features.children():
			feature_maps.append(module(feature_maps[-1]))
		return feature_maps


class Upsample(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(Upsample, self).__init__()
		self.convA = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
		self.leakyrelu = nn.LeakyReLU(0.2)
		self.convB = nn.Conv2d(output_channels, output_channels, 3, 1, 1)

	def forward(self, x, concat_with):
		concat_h_dim = concat_with.shape[2]
		concat_w_dim = concat_with.shape[3]

		upsampled_x = F.interpolate(x, size=[concat_h_dim, concat_w_dim], mode="bilinear", align_corners=True)
		upsampled_x = torch.cat([upsampled_x, concat_with], dim=1)
		upsampled_x = self.convA(upsampled_x)
		upsampled_x = self.leakyrelu(upsampled_x)
		upsampled_x = self.convB(upsampled_x)
		upsampled_x = self.leakyrelu(upsampled_x)
		
		return upsampled_x


class Decoder(nn.Module):
	def __init__(self, num_features=2208, decoder_width=0.5, scales=[1, 2, 4, 8]):
		super(Decoder, self).__init__()

		features = int(num_features * decoder_width)

		self.conv2 = nn.Conv2d(num_features, features, 1, 1, 0)
		self.upsample1 = Upsample(features // scales[0] + 384, features // (scales[0] * 2))
		self.upsample2 = Upsample(features // scales[1] + 192, features // (scales[1] * 2))
		self.upsample3 = Upsample(features // scales[2] + 96, features // (scales[2] * 2))
		self.upsample4 = Upsample(features // scales[3] + 96, features // (scales[3] * 2))
		self.conv3 = nn.Conv2d(features // (scales[3] * 2), 1, 3, 1, 1)

	def forward(self, features):
		x_block0 = features[3]
		x_block1 = features[4]
		x_block2 = features[6]
		x_block3 = features[8]
		x_block4 = features[11]

		x0 = self.conv2(x_block4)
		x1 = self.upsample1(x0, x_block3)
		x2 = self.upsample2(x1, x_block2)
		x3 = self.upsample3(x2, x_block1)
		x4 = self.upsample4(x3, x_block0)

		return self.conv3(x4)


# High Quality Monocular Depth Estimation via Transfer Learning
class DenseDepth(nn.Module):
	def __init__(self, encoder_pretrained=True):
		super(DenseDepth, self).__init__()

		self.encoder = Encoder(encoder_pretrained=encoder_pretrained)
		self.decoder = Decoder()

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x
	

if __name__ == '__main__':
	batch_size = 16
	device = ("cuda"
			  if torch.cuda.is_available()
			  else "mps"
			  if torch.backends.mps.is_available()
			  else "cpu")

	model = DenseDepth().to(device)
	summary(model, input_size=(batch_size, 3, 224, 224))
	
'''
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
DenseDepth                                         [16, 1, 112, 112]         --
├─Encoder: 1-1                                     [16, 3, 224, 224]         --
│    └─DenseNet: 2-1                               --                        2,209,000
│    │    └─Sequential: 3-1                        --                        26,472,000
├─Decoder: 1-2                                     [16, 1, 112, 112]         --
│    └─Conv2d: 2-2                                 [16, 1104, 7, 7]          2,438,736
│    └─Upsample: 2-3                               [16, 552, 14, 14]         --
│    │    └─Conv2d: 3-2                            [16, 552, 14, 14]         7,392,936
│    │    └─LeakyReLU: 3-3                         [16, 552, 14, 14]         --
│    │    └─Conv2d: 3-4                            [16, 552, 14, 14]         2,742,888
│    │    └─LeakyReLU: 3-5                         [16, 552, 14, 14]         --
│    └─Upsample: 2-4                               [16, 276, 28, 28]         --
│    │    └─Conv2d: 3-6                            [16, 276, 28, 28]         1,848,372
│    │    └─LeakyReLU: 3-7                         [16, 276, 28, 28]         --
│    │    └─Conv2d: 3-8                            [16, 276, 28, 28]         685,860
│    │    └─LeakyReLU: 3-9                         [16, 276, 28, 28]         --
│    └─Upsample: 2-5                               [16, 138, 56, 56]         --
│    │    └─Conv2d: 3-10                           [16, 138, 56, 56]         462,162
│    │    └─LeakyReLU: 3-11                        [16, 138, 56, 56]         --
│    │    └─Conv2d: 3-12                           [16, 138, 56, 56]         171,534
│    │    └─LeakyReLU: 3-13                        [16, 138, 56, 56]         --
│    └─Upsample: 2-6                               [16, 69, 112, 112]        --
│    │    └─Conv2d: 3-14                           [16, 69, 112, 112]        145,383
│    │    └─LeakyReLU: 3-15                        [16, 69, 112, 112]        --
│    │    └─Conv2d: 3-16                           [16, 69, 112, 112]        42,918
│    │    └─LeakyReLU: 3-17                        [16, 69, 112, 112]        --
│    └─Conv2d: 2-7                                 [16, 1, 112, 112]         622
====================================================================================================
Total params: 44,612,411
Trainable params: 44,612,411
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 258.82
====================================================================================================
Input size (MB): 9.63
Forward/backward pass size (MB): 5605.16
Params size (MB): 169.61
Estimated Total Size (MB): 5784.41
====================================================================================================
'''