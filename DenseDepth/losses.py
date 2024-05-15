import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional.image import image_gradients

def gaussian(window_size, sigma):
	"""Generate a Gaussian window normalized to sum to 1.
	
	Args:
		window_size (int): The size of the Gaussian window (one side).
		sigma (float): The standard deviation of the Gaussian.
		
	Returns:
		torch.Tensor: Normalized Gaussian window.
	"""
	gauss = torch.exp(-torch.pow(torch.arange(window_size) - window_size // 2, 2) / (2 * sigma ** 2))
	return gauss / gauss.sum()


def create_window(window_size, channel):
	"""Create a 2D Gaussian window used for SSIM calculation.
	
	Args:
		window_size (int): The diameter of the smoothing window.
		channel (int): Number of image channels.
	
	Returns:
		torch.Tensor: A 2D Gaussian window tensor replicated across channel dimension.
	"""
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
	return window


def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
	"""Calculate the mean Structural Similarity Index between two images.
	
	Args:
		img1, img2 (torch.Tensor): Images to compare; should be of shape (N, C, H, W)
		val_range (float): The dynamic range of the pixel values (e.g., 255 for 8-bit images)
		window_size (int, optional): Size of the smoothing window used in SSIM calculation.
		window (torch.Tensor, optional): Custom window for convolution; if None, defaults to Gaussian.
		size_average (bool, optional): If True, returns mean SSIM, else returns array of SSIM values.
		full (bool, optional): If True, also returns contrast metric alongside SSIM.
	
	Returns:
		float or tuple: The mean SSIM, or (mean SSIM, contrast metric) if `full` is True.
	"""
	if window is None:
		window = create_window(window_size, channel=img1.size(1)).to(img1.device)
		
	# Calculating the mu parameter (locally) for both images using a gaussian filter
	# Calculates the luminosity params
	mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
	mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
	
	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu12 = mu1 * mu2

	# Sigma deals with the contrast component
	sigma1_sq = F.conv2d(img1.pow(2), window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
	sigma2_sq = F.conv2d(img2.pow(2), window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
	sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu12

	# Some constants for stability
	C1 = (0.01 * val_range) ** 2
	C2 = (0.03 * val_range) ** 2
	ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
	
	if size_average:
		result = ssim_map.mean()
	else:
		result = ssim_map.mean([1, 2, 3])

	if full:
		contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
		return result, contrast_metric.mean()
	
	return result


def calculate_image_gradients(image_tensor, device):
	"""
	Calculate horizontal and vertical gradients for an image tensor.
	Calculates gradients by simple differences between adjacent pixelsadjacent pixels.

	Parameters:
	- image_tensor (torch.Tensor): The input tensor representing images of shape (batch_size, channels, height, width).
	- device (torch.device or str): The device to perform the operation on.

	Returns:
	- tuple[torch.Tensor, torch.Tensor]: A tuple of tensors (dy, dx) representing vertical and horizontal gradients.
	"""
	if image_tensor.ndim != 4:
		raise ValueError("Input tensor must be 4-dimensional (batch, channels, height, width)")

	batch_size, channels, height, width = image_tensor.shape

	vertical_gradient = image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :]
	horizontal_gradient = image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1]

	padded_vertical = torch.cat([
		vertical_gradient,
		torch.zeros(batch_size, channels, 1, width, device=device, dtype=image_tensor.dtype)
	], dim=2)

	padded_horizontal = torch.cat([
		horizontal_gradient,
		torch.zeros(batch_size, channels, height, 1, device=device, dtype=image_tensor.dtype)
	], dim=3)

	return padded_vertical, padded_horizontal


def calculate_image_gradients_sobel(image_tensor, device):
	"""
	Calculate horizontal and vertical gradients for an image tensor.
	Uses Sobel operators to compute gradients, which apply a weighted sum across a neighborhood and are more robust to noise.

	Parameters:
	- image_tensor (torch.Tensor): The input tensor representing images of shape (batch_size, channels, height, width).
	- device (torch.device or str): The device to perform the operation on.

	Returns:
	- tuple[torch.Tensor, torch.Tensor]: A tuple of tensors (dy, dx) representing vertical and horizontal gradients.
	"""

	# # Reverse the Sobel filters to align the gradient direction with the simple difference method
	# sobel_x = torch.tensor([[-1, 0, 1], 
	# 						[-2, 0, 2], 
	# 						[-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
	# sobel_y = torch.tensor([[-1, -2, -1], 
	# 						[0, 0, 0], 
	# 						[1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
	
	sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
	sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

	if image_tensor.shape[1] > 1:
		sobel_x = sobel_x.repeat(image_tensor.shape[1], 1, 1, 1)
		sobel_y = sobel_y.repeat(image_tensor.shape[1], 1, 1, 1)

	padding = 1
	horizontal_gradient = torch.nn.functional.conv2d(
		image_tensor, sobel_x, padding=padding, groups=image_tensor.shape[1])
	vertical_gradient = torch.nn.functional.conv2d(
		image_tensor, sobel_y, padding=padding, groups=image_tensor.shape[1])

	return vertical_gradient, horizontal_gradient


def compute_depth_loss(ground_truth, prediction, device="cuda", theta=0.1, maxDepth=1000.0 / 10.0):
	"""
	Compute the depth loss for predictions using image gradients and L1 loss.

	Parameters:
	- ground_truth (torch.Tensor): The ground truth depth image tensor.
	- prediction (torch.Tensor): The predicted depth image tensor.
	- device (str, optional): Computation device (default is 'cuda').
	- theta (float, optional): Weight for the L1 loss component (default is 0.1).

	Returns:
	- torch.Tensor: The computed loss.
	"""
	# Calculate gradients
	dy_true, dx_true = image_gradients(ground_truth)
	dy_pred, dx_pred = image_gradients(prediction)
	edge_loss = torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=1)

	return edge_loss


def main():
	# 【ssim()】
	import matplotlib.pyplot as plt
	from torchmetrics.functional.image import structural_similarity_index_measure
	from utils import load_image
	
	img1 = load_image('img_drone_1_RGB_0_1622304973074737100.png').to('cuda')
	img2 = load_image('img_drone_1_RGB_0_1622304972077115100.png').to('cuda')
	print(img1.shape) # Output: [1, 1, 480, 640]

	ssim_index = structural_similarity_index_measure(img1, img2, data_range=255)
	#ssim_index = ssim(img1, img2, val_range=255)
	print(f"SSIM Index: {ssim_index.item()}")
	
	# =========================================================
	# 【compute_depth_loss()】
	edge_loss = compute_depth_loss(img1, img2)
	print(f"Edge Loss: {torch.mean(edge_loss)}")
	
	# =========================================================
	# 【calculate_image_gradients()】
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
	test_image = load_image('001.jpg').to(device)
	print(test_image.shape)
	
	# Calculate gradients
	v_gradient, h_gradient = image_gradients(test_image)
	#v_gradient, h_gradient = calculate_image_gradients(test_image, device)
	#v_gradient, h_gradient = calculate_image_gradients_sobel(test_image, device)
	
	# Display results
	v_gradient = v_gradient.squeeze().permute(1, 2, 0).cpu().numpy()
	h_gradient = h_gradient.squeeze().permute(1, 2, 0).cpu().numpy()
	test_image = test_image.squeeze().permute(1, 2, 0).cpu().numpy()
	
	plt.figure(figsize=(12, 4))
	plt.subplot(1, 3, 1)
	plt.title('Original Image')
	plt.imshow(test_image)
	plt.axis('off')

	plt.subplot(1, 3, 2)
	plt.title('Vertical Gradient')
	plt.imshow(v_gradient)
	plt.axis('off')

	plt.subplot(1, 3, 3)
	plt.title('Horizontal Gradient')
	plt.imshow(h_gradient)
	plt.axis('off')

	plt.show()


if __name__ == "__main__":
	main()
