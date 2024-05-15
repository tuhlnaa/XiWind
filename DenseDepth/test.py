import cv2
import time
import torch
import pycuda.autoinit

import numpy as np
import seaborn as sns
import tensorrt as trt
import onnxruntime as ort
import torchvision.io as io
import pycuda.driver as cuda
import matplotlib.pyplot as plt


from typing import Union
from pathlib import Path
from enum import Enum, auto
from scipy.special import softmax
from torchvision import transforms
from functools import singledispatchmethod

from data import prepare_data_h5
from model import DenseDepth
from ESPADA.utils import load_image
from utils import compute_depth_estimation_metrics

class ONNXProvider(Enum):
	"""Enum to define supported ONNX runtime providers."""
	CUDA = auto()
	TENSORRT = auto()
	CPU = auto()

class TRTEngine:
	"""
	This class encapsulates the functionality for loading a TensorRT engine,
	allocating necessary resources, and performing inference.
	
	Attributes:
		engine (trt.ICudaEngine): The loaded TensorRT engine.
		context (trt.IExecutionContext): The execution context for the engine.
		inputs (list): A list of dictionaries containing the input buffer host and device pointers.
		outputs (list): A list of dictionaries containing the output buffer host and device pointers.
		bindings (list): A list of device buffer pointers required for engine execution.
		stream (cuda.Stream): The CUDA stream used for asynchronous operations.
	"""
	def __init__(self, engine_path):
		self.engine = self._load_engine(engine_path)
		self.context = self.engine.create_execution_context()
		self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

	def _load_engine(self, engine_path):
		"""
		Loads the TensorRT engine from a serialized file.
		
		Parameters:
			engine_path (str): The file path to the serialized TensorRT engine.
			
		Returns:
			trt.ICudaEngine: The loaded TensorRT engine.
		"""
		TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
		with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
			return runtime.deserialize_cuda_engine(f.read())

	def _allocate_buffers(self):
		"""
		Allocates necessary buffers for input and output based on the engine's requirements.
		
		Returns:
			tuple: A tuple containing lists of input and output buffers, bindings, and the CUDA stream.
		"""
		inputs, outputs, bindings = [], [], []
		stream = cuda.Stream()
		for binding in self.engine:
			# Adjust for explicit batch size handling
			shape = self.engine.get_tensor_shape(binding)
			dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
			size = trt.volume(shape)  # Assuming `shape` already accounts for batch size explicitly
			
			# Allocate host and device buffers
			host_mem = cuda.pagelocked_empty(size, dtype)
			device_mem = cuda.mem_alloc(host_mem.nbytes)
			bindings.append(int(device_mem))
			if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
				inputs.append({'host': host_mem, 'device': device_mem})
			else:
				outputs.append({'host': host_mem, 'device': device_mem})
		return inputs, outputs, bindings, stream

	def infer(self, input_data):
		"""
		Runs inference using the TensorRT engine with the provided input data.
		
		Parameters:
			input_data (np.ndarray): The input data to be processed by the engine.
			
		Returns:
			np.ndarray: The output data from the inference process.
		"""
		# Assuming a single input for simplicity. Extend as needed.
		np.copyto(self.inputs[0]['host'], input_data.ravel())  # Flatten input
		cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
		# Execute the model
		self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
		# Copy the output from device to host
		cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
		self.stream.synchronize()
		# Return the host output.
		return self.outputs[0]['host']


class BatchPredictor:
	"""
	A class to perform batch predictions using either PyTorch, ONNX models or TensorRT Engine.

	Attributes:
		model: The model to use for predictions. This can be a PyTorch or ONNX model.
		device: The device (CPU, CUDA, MPS) to run the model on.
		dataloader: A PyTorch DataLoader providing batches of images for prediction.
	"""
	
	def __init__(self, dataloader, device):
		self.device = device
		self.dataloader = dataloader

	@singledispatchmethod
	def predict_batch(self, model):
		raise NotImplementedError("Unsupported model type")

	@predict_batch.register
	def _(self, model: Union[torch.nn.Module, torch.jit.ScriptModule], save_outputs: bool = False):
		total_time = 0
		total_images = 0
		cumulative_metrics = {key: 0 for key in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10']}
		
		for data in self.dataloader:
			images, target = data["image"].to(self.device), data["depth"].to(self.device)
			
			start_time = time.monotonic()
			with torch.no_grad():
				outputs = model(images)
			total_time += time.monotonic() - start_time
			
			
			metrics = compute_depth_estimation_metrics(target, outputs)
			for key in cumulative_metrics:
				cumulative_metrics[key] += metrics[key]
			total_images += images.size(0)

			if save_outputs:
				# Save the outputs here, as images or tensors
				pass
		
		average_time = total_time / total_images
		for key in cumulative_metrics:
			cumulative_metrics[key] /= total_images

		return total_images, average_time, cumulative_metrics


	@predict_batch.register
	def _(self, model: ort.capi.onnxruntime_inference_collection.InferenceSession, save_outputs: bool = False):
		total_time = 0
		total_images = 0
		cumulative_metrics = {key: 0 for key in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10']}
	
		for data in self.dataloader:
			images, target = data["image"].numpy(), data["depth"]

			start_time = time.monotonic()
			outputs = model.run(None, {model.get_inputs()[0].name: images})
			total_time += time.monotonic() - start_time
			
			outputs = torch.tensor(outputs[0])
			print(outputs.shape)
			metrics = compute_depth_estimation_metrics(target, outputs)
			for key in cumulative_metrics:
				cumulative_metrics[key] += metrics[key]
			total_images += images.shape[0]

			if save_outputs:
				# Save the outputs here, as images or tensors
				pass
	
		average_time = total_time / total_images
		for key in cumulative_metrics:
			cumulative_metrics[key] /= total_images

		return total_images, average_time, cumulative_metrics


	@predict_batch.register
	def _(self, model: TRTEngine, save_outputs: bool = False):
		total_time = 0
		total_images = 0
		cumulative_metrics = {key: 0 for key in ['a1', 'a2', 'a3', 'abs_rel', 'rmse', 'log_10']}
	
		for data in self.dataloader:
			images, target = data["image"].numpy(), data["depth"]

			start_time = time.monotonic()
			outputs = model.infer(images)
			total_time += time.monotonic() - start_time

			outputs = np.reshape(outputs, (1, 1, int(images.shape[2]/2), int(images.shape[3]/2)))
			outputs = torch.tensor(outputs)
			metrics = compute_depth_estimation_metrics(target, outputs)
			for key in cumulative_metrics:
				cumulative_metrics[key] += metrics[key]
			total_images += images.shape[0]

			if save_outputs:
				# Save the outputs here, as images or tensors
				pass
	
		average_time = total_time / total_images
		for key in cumulative_metrics:
			cumulative_metrics[key] /= total_images

		return total_images, average_time, cumulative_metrics


class ImagePredictor:
	"""
	A class to perform single image predictions using either PyTorch or ONNX models.
 
	Attributes:
		device: The device (CPU, CUDA, MPS) to run the model on.
		img_tensor: The tensor representing the preprocessed image.
	"""
	def __init__(self, img_path, device):
		self.device = device
		self.img_tensor = self._preprocess_image(img_path).to(device)


	def _preprocess_image(self, img_path):
		"""
		Preprocesses the image from the given path.
		
		Args:
			img_path: The path to the image file.
 
		Returns:
			A tensor representing the processed image, ready for model input.
		"""
		img = cv2.imread(str(img_path))
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		preprocess = transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor()])
		img_tensor = preprocess(img_rgb)
		return img_tensor.unsqueeze(0)


	@singledispatchmethod
	def predict_image(self, model):
		"""
		Raises:
			NotImplementedError: If the model type is unsupported.
		"""
		raise NotImplementedError("Unsupported model type")


	@predict_image.register
	def _(self, model: Union[torch.nn.Module, torch.jit.ScriptModule]):
		"""
		Predicts the class for a single image using a PyTorch model.
 
		Args:
			model: A PyTorch model (either nn.Module or ScriptModule) for prediction.
 
		Returns:
			A tensor representing the model's output.
		"""
		start_time = time.monotonic()
		with torch.no_grad():
			outputs = model(self.img_tensor)
		elapsed_time = time.monotonic() - start_time
		
		outputs = outputs.squeeze(0).to('cpu') * 255
		outputs = outputs.to(torch.uint8)
		return outputs
	

	@predict_image.register
	def _(self, model: ort.capi.onnxruntime_inference_collection.InferenceSession):
		"""
		Predict the depth for a single image using an ONNX model.

		Args:
			model: An ONNX InferenceSession object for prediction.

		Returns:
			A tensor representing the model's output.
		"""
		img = self.img_tensor.cpu().numpy()
		
		start_time = time.monotonic()
		outputs = model.run(None, {model.get_inputs()[0].name: img})
		elapsed_time = time.monotonic() - start_time

		outputs = torch.tensor(outputs[0]).squeeze(0).to('cpu') * 255
		outputs = outputs.to(torch.uint8)
		return outputs


	@predict_image.register
	def _(self, model: TRTEngine):
		"""
		Predict the depth for a single image using a TensorRT engine.
		
		Args:
			model: A TRTEngine object for prediction.

		Returns:
			A tensor representing the model's output.
		"""
		img = self.img_tensor.cpu().numpy()

		start_time = time.monotonic()
		outputs = model.infer(img)
		elapsed_time = time.monotonic() - start_time

		outputs = np.reshape(outputs, (1, int(img.shape[2]/2), int(img.shape[3]/2)))
		outputs = torch.tensor(outputs).to('cpu') * 255
		outputs = outputs.to(torch.uint8)
		return outputs
	

class ModelLoader:
	"""
	A class for loading machine learning models, supporting both PyTorch and ONNX formats.

	Attributes:
		model_path: The file path of the model to load.
		device: The device (CPU, CUDA, MPS) to load the model onto.
		provider_preference: Preferred ONNX runtime provider for ONNX models.
	"""
	def __init__(self, model_path, device, provider_preference=ONNXProvider.CPU):
		self.model_path = str(model_path)
		self.device = device
		self.provider_preference = provider_preference


	def load(self):
		if self.model_path.endswith('.onnx'):
			return self._load_onnx_model()
		elif self.model_path.endswith('.trt'):
			return self._load_tensorrt_engine()
		else:
			return self._load_pytorch_model()


	def _load_onnx_model(self):
		"""
		Loads an ONNX model with specified runtime providers.

		Returns:
			An ONNX InferenceSession with the loaded model.
		"""
		providers = ['CPUExecutionProvider']
		if torch.cuda.is_available():
			if self.provider_preference == ONNXProvider.CUDA:
				providers.insert(0, 'CUDAExecutionProvider')
			elif self.provider_preference == ONNXProvider.TENSORRT:
				providers.insert(0, 'TensorrtExecutionProvider')

		sess_options = ort.SessionOptions()
		sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
		return ort.InferenceSession(self.model_path, sess_options, providers=providers)


	def _load_tensorrt_engine(self):
		"""
		Loads a TensorRT engine for inference.

		Returns:
			An instance of TRTEngine with the loaded engine.
		"""
		return TRTEngine(self.model_path)
	

	def _load_pytorch_model(self):
		"""
		Loads a PyTorch model (.pth or .pt file) for inference.

		Returns:
			The loaded PyTorch model.

		Raises:
			ValueError: If the model file format is not .pth or .pt.
		"""
		if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
			model = self._load_standard_or_scripted_pytorch_model()
		else:
			raise ValueError("Unsupported model file format.")
		model.eval()
		return model


	def _load_standard_or_scripted_pytorch_model(self):
		"""
		Helper method to load standard or scripted PyTorch models.

		Returns:
			The loaded PyTorch model, optimized for inference if applicable.
		"""
		if self.model_path.endswith('.pth'):
			model = DenseDepth().to(self.device)
			checkpoint = torch.load(self.model_path, map_location=self.device)
			model.load_state_dict(checkpoint['model_state_dict'])
		elif self.model_path.endswith('.pt'):
			model = torch.jit.load(self.model_path, map_location=self.device)
			model = torch.jit.optimize_for_inference(model)
		return model


def main():
	batch_size = 1   # Batch size
	
	root_dir = Path(__file__).parent
	test_data_dir = root_dir / "dataset" / "test"
	image_path = root_dir / "01335_colors.png"
	
	model_path = root_dir / 'checkpoint' / 'model_epoch_30.pth'                   # Torch model
	model_path = root_dir / 'inference_model' / "DenseDepth_scripted_model.pt"    # Torch Script model
	model_path = root_dir / 'inference_model' / "DenseDepth_scripted_model.onnx"  # ONNX model
	model_path = root_dir / 'inference_model' / "DenseDepth_scripted_model.trt"   # TensorRT engine
	
	test_data_dir = (root_dir / "dataset/nyudepthv2/train", 
					 root_dir / "dataset/nyudepthv2/val")
	
	device = ("cuda"
			  if torch.cuda.is_available() 
			  else "mps"
			  if torch.backends.mps.is_available()
			  else "cpu")

	# Loading
	_, test_loader = prepare_data_h5(test_data_dir, batch_size=batch_size)
	
	image_predictor = ImagePredictor(image_path, device)
	batch_predictor = BatchPredictor(test_loader, device)
	
	model_loader = ModelLoader(model_path, device, provider_preference=ONNXProvider.CUDA)
	model = model_loader.load()
	
	# Single Image Prediction
	print('[Single Image Prediction]')
	outputs = image_predictor.predict_image(model)
	io.write_png(outputs, "output.png")
	
	# Batch Prediction
	print('[Batch Prediction]')
	total_images, avg_time, cumulative_metrics = batch_predictor.predict_batch(model)
	print(f"total images: {total_images}, Average time per image: {avg_time:.6f} seconds")
	print(cumulative_metrics)
	

if __name__ == "__main__":
	main()


'''
Average time per image, Python 3.11, torch 2.2.0+cu121
[GTX1080 Ti] R7 5700X
PyTorch (CUDA 12.1): 24ms
TorchScript (CUDA 12.1): 86~95ms
ONNX (CUDA 11.8): 50~52ms
TensorRT (CUDA 11.8): 35ms
'''