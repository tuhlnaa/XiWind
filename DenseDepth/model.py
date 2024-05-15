"""
High Quality Monocular Depth Estimation via Transfer Learning
https://arxiv.org/abs/1812.11941
"""

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import List
from torchinfo import summary
from torchvision.models import densenet161, DenseNet161_Weights

class Encoder(nn.Module):
	def __init__(self, encoder_pretrained=DenseNet161_Weights.IMAGENET1K_V1):
		super(Encoder, self).__init__()
		self.densenet = densenet161(weights=encoder_pretrained)

	def forward(self, x: torch.Tensor):
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

	def forward(self, features: List[torch.Tensor]):
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


class DenseDepth(nn.Module):
	def __init__(self, encoder_pretrained=DenseNet161_Weights.IMAGENET1K_V1):
		super(DenseDepth, self).__init__()

		self.encoder = Encoder(encoder_pretrained=encoder_pretrained)
		self.decoder = Decoder()

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


def load_model_checkpoint(model, checkpoint_path, device):
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])


def convert_to_torchscript(model, scripted_model_path, device):
	model.eval()
	scripted_model = torch.jit.script(model)
	scripted_model.save(scripted_model_path)


def convert_to_onnx(model, onnx_model_path, device):
	model.eval()
	x = torch.rand((1, 3, 480, 640), dtype=torch.float32).to(device)
	torch.onnx.export(
		model,
		x,
		onnx_model_path,
		export_params=True,
		opset_version=17,
		do_constant_folding=True,
		input_names=['input'],
		output_names=['output'],
		dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
	)
	onnx_model = onnx.load(onnx_model_path)
	onnx.checker.check_model(onnx_model)


def main():
	batch_size = 8
	device = ("cuda"
			  if torch.cuda.is_available()
			  else "mps"
			  if torch.backends.mps.is_available()
			  else "cpu")

	model = DenseDepth().to(device)
	summary(model, input_size=(batch_size, 3, 480, 640))

	root_dir = Path(__file__).parent
	checkpoint_path = root_dir / 'checkpoint' / 'model_epoch_30.pth'
	load_model_checkpoint(model, checkpoint_path, device)

	scripted_model_path = root_dir / 'inference_model' / 'DenseDepth_scripted_model.pt'
	convert_to_torchscript(model, scripted_model_path, device)

	onnx_model_path = root_dir / 'inference_model' / 'DenseDepth_scripted_model.onnx'
	convert_to_onnx(model, onnx_model_path, device)
	
	# Convert to TensorRT
	# trtexec --onnx=DenseDepth_scripted_model.onnx --saveEngine=DenseDepth_scripted_model.trt --minShapes=input:1x3x480x640 --optShapes=input:1x3x480x640 --maxShapes=input:1x3x480x640 --buildOnly 


if __name__ == '__main__':
	main()

	
'''
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
DenseDepth                                         [8, 1, 240, 320]          --
├─Encoder: 1-1                                     [8, 3, 480, 640]          --
│    └─DenseNet: 2-1                               --                        2,209,000
│    │    └─Sequential: 3-1                        --                        26,472,000
├─Decoder: 1-2                                     [8, 1, 240, 320]          --
│    └─Conv2d: 2-2                                 [8, 1104, 15, 20]         2,438,736
│    └─Upsample: 2-3                               [8, 552, 30, 40]          --
│    │    └─Conv2d: 3-2                            [8, 552, 30, 40]          7,392,936
│    │    └─LeakyReLU: 3-3                         [8, 552, 30, 40]          --
│    │    └─Conv2d: 3-4                            [8, 552, 30, 40]          2,742,888
│    │    └─LeakyReLU: 3-5                         [8, 552, 30, 40]          --
│    └─Upsample: 2-4                               [8, 276, 60, 80]          --
│    │    └─Conv2d: 3-6                            [8, 276, 60, 80]          1,848,372
│    │    └─LeakyReLU: 3-7                         [8, 276, 60, 80]          --
│    │    └─Conv2d: 3-8                            [8, 276, 60, 80]          685,860
│    │    └─LeakyReLU: 3-9                         [8, 276, 60, 80]          --
│    └─Upsample: 2-5                               [8, 138, 120, 160]        --
│    │    └─Conv2d: 3-10                           [8, 138, 120, 160]        462,162
│    │    └─LeakyReLU: 3-11                        [8, 138, 120, 160]        --
│    │    └─Conv2d: 3-12                           [8, 138, 120, 160]        171,534
│    │    └─LeakyReLU: 3-13                        [8, 138, 120, 160]        --
│    └─Upsample: 2-6                               [8, 69, 240, 320]         --
│    │    └─Conv2d: 3-14                           [8, 69, 240, 320]         145,383
│    │    └─LeakyReLU: 3-15                        [8, 69, 240, 320]         --
│    │    └─Conv2d: 3-16                           [8, 69, 240, 320]         42,918
│    │    └─LeakyReLU: 3-17                        [8, 69, 240, 320]         --
│    └─Conv2d: 2-7                                 [8, 1, 240, 320]          622
====================================================================================================
Total params: 44,612,411
Trainable params: 44,612,411
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 792.28
====================================================================================================
Input size (MB): 29.49
Forward/backward pass size (MB): 17158.66
Params size (MB): 169.61
Estimated Total Size (MB): 17357.76
====================================================================================================
'''