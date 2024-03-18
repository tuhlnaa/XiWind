import os
import onnx
import torch
from torch import nn
from torchinfo import summary
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Create the new model by using MobileNetV2 as the based neural network architecture with the pretrained model from ImageNet. 
# This includes Kaiming intialization.
class DRModel(nn.Module):
	def __init__(self, weights=MobileNet_V2_Weights.DEFAULT):
		super(DRModel, self).__init__()
		if weights is not None:
			backbone = mobilenet_v2(weights=weights)
		else:
			backbone = mobilenet_v2(weights=None)

		self.backbone = backbone.features
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(0.2)
		self.classifier = nn.Linear(backbone.last_channel, 2)

		if weights is None:
			self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x = self.backbone(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.dropout(x)
		return self.classifier(x)

if __name__ == '__main__':
	batch_size = 32
	device = ("cuda"
			  if torch.cuda.is_available()
			  else "mps"
			  if torch.backends.mps.is_available()
			  else "cpu")

	model = DRModel(weights=MobileNet_V2_Weights.IMAGENET1K_V2).to(device)
	summary(model, input_size=(batch_size, 3, 224, 224))
	# =========================================================

	# Example usage of save and load functions
	#torch.save(model.state_dict(), "model.pth")
	#model.load_state_dict(torch.load("model.pth", map_location='cpu'))
	
	# =========================================================
	# Load the saved model checkpoint
	checkpoint_path = 'model_epoch_40.pth'
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint['model_state_dict'])

	# Convert to TorchScript
	scripted_model_path = 'MobileNetV2_scripted_model.pt'
	scripted_model = torch.jit.script(model)
	scripted_model.save(scripted_model_path)

	# Convert to ONNX
	model.eval()
	x = torch.rand((1, 3, 224, 224), dtype=torch.float32).to(device)
	torch.onnx.export(
		model,
		x,
		"MobileNetV2_model.onnx",
		export_params=True,
		opset_version=17,
		do_constant_folding=True,
		input_names=['input'],
		output_names=['output'],
		dynamic_axes={'input' : {0 : 'batch_size'},
					  'output' : {0 : 'batch_size'}})
	onnx_model = onnx.load("model.onnx")
	onnx.checker.check_model(onnx_model)

	# Convert to TensorRT
	# trtexec --onnx=MobileNetV2_model.onnx --saveEngine=MobileNetV2_model.trt --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:1x3x224x224 --buildOnly 

	

"""
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
DR_model                                           [32, 2]                   --
├─Sequential: 1-1                                  [32, 1280, 7, 7]          --
│    └─Conv2dNormActivation: 2-1                   [32, 32, 112, 112]        --
│    │    └─Conv2d: 3-1                            [32, 32, 112, 112]        864
│    │    └─BatchNorm2d: 3-2                       [32, 32, 112, 112]        64
│    │    └─ReLU6: 3-3                             [32, 32, 112, 112]        --
│    └─InvertedResidual: 2-2                       [32, 16, 112, 112]        --
│    │    └─Sequential: 3-4                        [32, 16, 112, 112]        896
│    └─InvertedResidual: 2-3                       [32, 24, 56, 56]          --
│    │    └─Sequential: 3-5                        [32, 24, 56, 56]          5,136
│    └─InvertedResidual: 2-4                       [32, 24, 56, 56]          --
│    │    └─Sequential: 3-6                        [32, 24, 56, 56]          8,832
│    └─InvertedResidual: 2-5                       [32, 32, 28, 28]          --
│    │    └─Sequential: 3-7                        [32, 32, 28, 28]          10,000
│    └─InvertedResidual: 2-6                       [32, 32, 28, 28]          --
│    │    └─Sequential: 3-8                        [32, 32, 28, 28]          14,848
│    └─InvertedResidual: 2-7                       [32, 32, 28, 28]          --
│    │    └─Sequential: 3-9                        [32, 32, 28, 28]          14,848
│    └─InvertedResidual: 2-8                       [32, 64, 14, 14]          --
│    │    └─Sequential: 3-10                       [32, 64, 14, 14]          21,056
│    └─InvertedResidual: 2-9                       [32, 64, 14, 14]          --
│    │    └─Sequential: 3-11                       [32, 64, 14, 14]          54,272
│    └─InvertedResidual: 2-10                      [32, 64, 14, 14]          --
│    │    └─Sequential: 3-12                       [32, 64, 14, 14]          54,272
│    └─InvertedResidual: 2-11                      [32, 64, 14, 14]          --
│    │    └─Sequential: 3-13                       [32, 64, 14, 14]          54,272
│    └─InvertedResidual: 2-12                      [32, 96, 14, 14]          --
│    │    └─Sequential: 3-14                       [32, 96, 14, 14]          66,624
│    └─InvertedResidual: 2-13                      [32, 96, 14, 14]          --
│    │    └─Sequential: 3-15                       [32, 96, 14, 14]          118,272
│    └─InvertedResidual: 2-14                      [32, 96, 14, 14]          --
│    │    └─Sequential: 3-16                       [32, 96, 14, 14]          118,272
│    └─InvertedResidual: 2-15                      [32, 160, 7, 7]           --
│    │    └─Sequential: 3-17                       [32, 160, 7, 7]           155,264
│    └─InvertedResidual: 2-16                      [32, 160, 7, 7]           --
│    │    └─Sequential: 3-18                       [32, 160, 7, 7]           320,000
│    └─InvertedResidual: 2-17                      [32, 160, 7, 7]           --
│    │    └─Sequential: 3-19                       [32, 160, 7, 7]           320,000
│    └─InvertedResidual: 2-18                      [32, 320, 7, 7]           --
│    │    └─Sequential: 3-20                       [32, 320, 7, 7]           473,920
│    └─Conv2dNormActivation: 2-19                  [32, 1280, 7, 7]          --
│    │    └─Conv2d: 3-21                           [32, 1280, 7, 7]          409,600
│    │    └─BatchNorm2d: 3-22                      [32, 1280, 7, 7]          2,560
│    │    └─ReLU6: 3-23                            [32, 1280, 7, 7]          --
├─AdaptiveAvgPool2d: 1-2                           [32, 1280, 1, 1]          --
├─Dropout: 1-3                                     [32, 1280]                --
├─Linear: 1-4                                      [32, 2]                   2,562
====================================================================================================
Total params: 2,226,434
Trainable params: 2,226,434
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 9.58
====================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 3419.19
Params size (MB): 8.91
Estimated Total Size (MB): 3447.37
====================================================================================================
"""