"""Training method for High Quality Monocular Depth Estimation"""

import time
import cv2
import torch
import torchvision.utils as vision_utils

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.image import structural_similarity_index_measure

from data import prepare_data_tensorflow
from model import DenseDepth
from losses import ssim
from losses import compute_depth_loss
from utils import AverageMeter, depth_norm, apply_colormap, load_from_checkpoint, init_or_load_model

import numpy as np
import matplotlib.pyplot as plt

class Logger:
	def __init__(self, log_dir, checkpoint_dir):
		self.writer = SummaryWriter(log_dir)
		self.checkpoint_dir = checkpoint_dir
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
			
	def log_scalars(self, tag, values, epoch):
		self.writer.add_scalars(tag, values, epoch)
		
	def add_image(self, tag, img_tensor, epoch):
		self.writer.add_image(tag, img_tensor, global_step=epoch, dataformats='CHW')

	def close(self):
		self.writer.close()
		
	def save_checkpoint(self, epoch, model, optimizer, loss, frequency=3):
		if (epoch + 1) % frequency == 0:
			checkpoint_path = self.checkpoint_dir / f'model_epoch_{epoch+1}.pth'
			torch.save({
				'epoch': epoch+1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				#'scheduler_state_dict': scheduler.state_dict(),
				'last_loss': loss,
			}, checkpoint_path)
			print(f"Checkpoint saved at '{checkpoint_path}'")

	def load_checkpoint(self, model, optimizer):
		checkpoints = list(self.checkpoint_dir.glob('model_epoch_*.pth'))
		if checkpoints:
			# Extract epochs from file names and find the max epoch
			latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
			checkpoint = torch.load(latest_checkpoint)
			model.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			#scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			epoch = checkpoint['epoch']
			loss = checkpoint['last_loss']
			print(f"Loaded checkpoint from '{latest_checkpoint}' at epoch {epoch} with loss {loss}")
			return epoch, loss
		else:
			print("No checkpoints found.")
			return 0, None  # Return defaults if no checkpoint found


# Training function
def train(model, device, train_loader, optimizer, criterion, theta):
	model.train()
	total_loss, total_samples = 0, 0
	
	for batch_idx, data in enumerate(train_loader):
		images, target = data["image"].to(device), data["depth"].to(device)
		normalized_target = depth_norm(target)
		optimizer.zero_grad()     # Reset gradients
		output = model(images)	  # Forward pass
		
		# Compute loss
		loss_l1 = criterion(output, normalized_target)  
		loss_ssim = torch.clamp(
			(1 - structural_similarity_index_measure(output, 1 / normalized_target, data_range=1)) * 0.5, 
			#(1 - ssim(output, normalized_target, 1000.0 / 10.0)) * 0.5, 
			min=0, max=1)
		loss_gradient = compute_depth_loss(1 / normalized_target, output, device=device)
		loss_combined = ((theta * torch.mean(loss_l1)) + 
						 (1.0 * loss_ssim) + 
						 (0.1 * torch.mean(loss_gradient)))
		
		loss_combined.backward()  # Backward pass
		optimizer.step()		  # Update parameters

		total_samples += len(images)
		total_loss += loss_combined.item()
		
	return total_loss / total_samples


# Validation function 
def validate(model, device, loader, criterion, epoch, theta, logger):
	model.eval()
	total_loss, total_samples = 0, 0

	with torch.no_grad():
		for batch_idx, data in enumerate(loader):
			images, target = data["image"].to(device), data["depth"].to(device)
			normalized_target = depth_norm(target)
			output = model(images)
			
			# Compute loss
			loss_l1 = criterion(output, normalized_target)  
			loss_ssim = torch.clamp(
				(1 - structural_similarity_index_measure(1 / normalized_target, output, data_range=1)) * 0.5, 
				min=0, max=1)
			loss_gradient = compute_depth_loss(1 / normalized_target, output, device=device)
			loss_combined = ((theta * torch.mean(loss_l1)) + 
							 (1.0 * loss_ssim) + 
							 (1.0 * torch.mean(loss_gradient)))

			total_samples += len(images)
			total_loss += loss_combined.item()

			if batch_idx == 0:  # Additional logging for the first batch
				output = torch.clamp(output, 0, 1)
				logger.add_image(
					"Model Outputs", 
					vision_utils.make_grid(apply_colormap(output), nrow=6, normalize=False), 
					epoch)
				logger.add_image(
					"Different", 
					vision_utils.make_grid(apply_colormap(torch.abs(output - (1 / normalized_target))), nrow=6, normalize=False), 
					epoch)	
				
		# Visualize the first batch of images and results at the start of validate	
		if epoch == 0:
			data = next(iter(loader))
			images, target = data["image"].to(device), data["depth"].to(device)
			logger.add_image(
				"Images", 
				vision_utils.make_grid(images, nrow=6, normalize=False),
				epoch)
			logger.add_image(
				"Target Depth Maps", 
				vision_utils.make_grid(apply_colormap(target / 1000), nrow=6, normalize=False),
				epoch)

	return total_loss / total_samples		
	

def main():
	# Define configuration parameters directly
	epochs = 27                # number of epochs for training
	lr = 0.0001                # initial learning rate
	batch_size = 4             # Batch size
	checkpoint = ""            # path to last saved checkpoint (empty string indicates none)
	resume_epoch = -1          # epoch to resume training (-1 indicates start from beginning)
	device = "cuda"            # device to run training ('cuda' or 'cpu')
	encoder_pretrained = True  # Use pretrained encoder (True or False)
	data_path = "data/"        # path to dataset
	theta = 0.1                # coeff for L1 (depth) Loss
	save_path = ""             # location to save checkpoints in (empty string indicates default location)
	
	root_dir = Path(__file__).parent
	checkpoint_dir = root_dir / "checkpoint"
	rgb_dir = (root_dir / "dataset/ESPADA_png_RGB/ESPADA_PH/TRAIN", 
			   root_dir / "dataset/ESPADA_png_RGB/ESPADA_PH/VAL")
	depth_dir = (root_dir / "dataset/ESPADA_depth_npy/ESPADA_PH/TRAIN", 
				 root_dir / "dataset/ESPADA_depth_npy/ESPADA_PH/VAL")
	dir = (root_dir / "dataset/nyudepthv2/train", 
		   root_dir / "dataset/nyudepthv2/val")
	logger = Logger('runs/experiment_DenseDepth', checkpoint_dir)

	# Training utils
	batch_size = batch_size
	model_prefix = "densedepth_"
	device = ("cuda"
			  if torch.cuda.is_available() 
			  else "mps"
			  if torch.backends.mps.is_available()
			  else "cpu")
	save_count = 0
	epoch_loss = []
	batch_loss = []
	sum_loss = 0
	train_loader, test_loader = prepare_data_tensorflow(dir, batch_size=batch_size)
	
	# Model, Optimizer, Scheduler, Loss function initialization
	model = DenseDepth(encoder_pretrained=encoder_pretrained).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.L1Loss()
	
	# Load checkpoint if exists
	start_epoch, _ = logger.load_checkpoint(model, optimizer)
		
	for epoch in range(start_epoch, epochs):
		train_loss = train(model, device, train_loader, optimizer, criterion, theta)
		get_last_lr = optimizer.state_dict()['param_groups'][0]['lr']
		print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Learning Rate: {get_last_lr}, {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
		
		validation_loss = validate(model, device, test_loader, criterion, epoch, theta, logger)
		print(f'Validation Loss: {validation_loss:.4f}\n')
		
		logger.log_scalars('Loss', {'Training': train_loss, 'Validation': validation_loss}, epoch)
		logger.save_checkpoint(epoch, model, optimizer, train_loss)
	logger.close()


if __name__ == "__main__":
	main()