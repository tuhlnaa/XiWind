import cv2
import h5py
import math
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import IterableDataset, DataLoader, Dataset


class DepthDataset(Dataset):
	def __init__(self, rgb_dir, depth_dir, transform=None):
		"""
		Args:
			rgb_dir (string): Directory with all the RGB images.
			depth_dir (string): Directory with all the depth files.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.rgb_dir = rgb_dir
		self.depth_dir = depth_dir
		self.transform = transform

		# Recursively list all files in the respective directories
		rgb_files = sorted([x for x in self.rgb_dir.rglob('*_RGB_*.png') if x.is_file()])
		depth_files = sorted([x for x in self.depth_dir.rglob('*_depth_*.npy') if x.is_file()])

		# Filter out files and create a mapping based on the filenames
		self.file_pairs = []
		for i in range(len(rgb_files)):
			self.file_pairs.append((rgb_files[i], depth_files[i]))

	def __len__(self):
		return len(self.file_pairs)

	def __getitem__(self, idx):
		rgb_path, depth_path = self.file_pairs[idx]
		image = Image.open(rgb_path).convert('RGB')
		depth = np.load(depth_path)
		#depth = Image.open(depth_path).convert('L')  # Assuming depth images are grayscale

		sample = {'image': image, 'depth': depth}

		if self.transform:
			sample = self.transform(sample)

		return sample
	

def load_file_pairs(rgb_dir, depth_dir):
	"""
	Generator function to yield RGB and depth file pairs.
	"""
	rgb_dir = rgb_dir
	depth_dir = depth_dir

	#rgb_files = sorted([x for x in rgb_dir.rglob('*_RGB_*.png') if x.is_file()])
	#depth_files = sorted([x for x in depth_dir.rglob('*_depth_*.npy') if x.is_file()])
	
	rgb_files = sorted([x for x in rgb_dir.rglob('*.jpg') if x.is_file()])
	depth_files = sorted([x for x in depth_dir.rglob('*.png') if x.is_file()])
	
	for rgb_file, depth_file in zip(rgb_files, depth_files):
		yield rgb_file, depth_file


class DepthIterableDataset_NYU_Kaggle(IterableDataset):
	def __init__(self, rgb_dir, depth_dir, transform=None, shuffle=True):
		"""
		Args:
			rgb_dir (string): Directory with all the RGB images.
			depth_dir (string): Directory with all the depth files.
			transform (callable, optional): Optional transform to be applied on a sample.
			shuffle (bool, optional): Whether to shuffle the data, default is True.
		"""
		self.rgb_dir = rgb_dir
		self.depth_dir = depth_dir
		self.transform = transform
		self.shuffle = shuffle
		self.file_pairs = list(load_file_pairs(rgb_dir, depth_dir))

	def __iter__(self):
		total_size = len(self.file_pairs)
		worker_info = torch.utils.data.get_worker_info()
		start, end = 0, total_size  # Default for single-process
		
		# In a worker process, split the workload
		if worker_info is not None:
			per_worker = int(math.ceil(total_size / float(worker_info.num_workers)))
			worker_id = worker_info.id
			start = worker_id * per_worker
			end = min(start + per_worker, total_size)
			
		# Shuffle the subset for each worker if shuffle is True
		subset_file_pairs = self.file_pairs[start:end]
		if self.shuffle:
			random.shuffle(subset_file_pairs)
		
		for rgb_path, depth_path in subset_file_pairs:
			image = Image.open(rgb_path).convert('RGB')
			depth = np.array(Image.open(depth_path).convert('L'))
			#depth = np.load(depth_path)
			sample = {'image': image, 'depth': depth}
			if self.transform:
				sample = self.transform(sample)
			yield sample
		

class DepthIterableDataset_NYU_TensorFlow(IterableDataset):
	def __init__(self, dir, transform=None, shuffle=True):
		"""
		Args:
			dir (string): Directory with all the RGB images and depth files.
			transform (callable, optional): Optional transform to be applied on a sample.
			shuffle (bool, optional): Whether to shuffle the data, default is True.
		"""
		self.dir = dir
		self.transform = transform
		self.shuffle = shuffle
		self.file_pairs = list(sorted([x for x in dir.rglob('*.h5') if x.is_file()]))

	def __iter__(self):
		total_size = len(self.file_pairs)
		worker_info = torch.utils.data.get_worker_info()
		start, end = 0, total_size  # Default for single-process
		
		# In a worker process, split the workload
		if worker_info is not None:
			per_worker = int(math.ceil(total_size / float(worker_info.num_workers)))
			worker_id = worker_info.id
			start = worker_id * per_worker
			end = min(start + per_worker, total_size)
			
		# Shuffle the subset for each worker if shuffle is True
		subset_file_pairs = self.file_pairs[start:end]
		if self.shuffle:
			random.shuffle(subset_file_pairs)

		for path in subset_file_pairs:
			# Open the H5 file for reading
			with h5py.File(path, 'r') as file:
				keys = list(file.keys())
				#print(f"Keys: {file.keys()}") # List all groups
				
				# Get the data (Change keys[index] based on your dataset structure)
				depth = np.array(file[keys[0]])
				image = np.array(file[keys[1]])
				image = np.transpose(image, (1, 2, 0))  # Change (C, H, W) to (H, W, C)

				sample = {'image': image, 'depth': depth}
				if self.transform:
					sample = self.transform(sample)
				yield sample


class RandomHorizontalFlip():
	def __init__(self, p=0.5):
		self.probability = p
		
	def __call__(self, sample):
		image, depth = sample['image'], sample['depth']
		if random.random() < self.probability:
			image = transforms.functional.hflip(image)
			depth = transforms.functional.hflip(depth)
		return {'image': image, 'depth': depth}


class RandomChannelSwap():
	def __init__(self, probability=0.5):
		self.probability = probability

	def __call__(self, sample):
		image = sample['image']
		if random.random() < self.probability:
			channels = [0, 1, 2]
			random.shuffle(channels)
			image = image[channels,:,:]
		return {'image': image, 'depth': sample['depth']}


class ToTensor():
	def __init__(self, is_test=False):
		self.is_test = is_test
		
	def __call__(self, sample):
		image, depth = sample['image'], sample['depth']
		depth = cv2.resize(depth, (320, 240))
		
		image = transforms.functional.to_tensor(image)
		depth = transforms.functional.to_tensor(depth)
		
		# Depth Norm (maxDepth=1000cm)
		depth = (depth - 0) / (10 - 0)
		if self.is_test:
			depth = depth / 1000
		else:            
			depth = depth * 1000
		depth = torch.clamp(depth, 10, 1000)  # Ensure depth is in expected range
		#depth = 1000 / depth
		#depth = torch.clamp(depth, 0, 1).to(torch.float32)  # Ensure depth is in expected range
		
		return {'image': image, 'depth': depth}


def get_transforms(is_train=True):
	transforms_list = [ToTensor()]  # ToTensor(is_test=not is_train)
	if is_train:
		transforms_list.insert(1, RandomHorizontalFlip(p=0.5))
		transforms_list.insert(2, RandomChannelSwap(0.5))
	return transforms.Compose(transforms_list)
	

def prepare_data_kaggle(rgb_dir, depth_dir, batch_size=8, num_workers=1):
	'''Load and Preprocess Dataset'''
	# Initialize datasets
	train_dataset = DepthIterableDataset_NYU_Kaggle(rgb_dir=rgb_dir[0], depth_dir=depth_dir[0], transform=get_transforms(is_train=True), shuffle=True)
	test_dataset = DepthIterableDataset_NYU_Kaggle(rgb_dir=rgb_dir[1], depth_dir=depth_dir[1], transform=get_transforms(is_train=False), shuffle=False)

	# Initialize data loaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_loader, test_loader


def prepare_data_tensorflow(dir, batch_size=8, num_workers=1):
	'''Load and Preprocess Dataset'''
	# Initialize datasets
	train_dataset = DepthIterableDataset_NYU_TensorFlow(dir=dir[0], transform=get_transforms(is_train=True), shuffle=True)
	test_dataset = DepthIterableDataset_NYU_TensorFlow(dir=dir[1], transform=get_transforms(is_train=False), shuffle=False)

	# Initialize data loaders
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return train_loader, test_loader


if __name__ == "__main__":
	root_dir = Path(__file__).parent
	# rgb_dir = (root_dir / "dataset/ESPADA_png_RGB/ESPADA_PH/TRAIN", 
	# 		   root_dir / "dataset/ESPADA_png_RGB/ESPADA_PH/VAL")
	# depth_dir = (root_dir / "dataset/ESPADA_depth_npy/ESPADA_PH/TRAIN", 
	# 			 root_dir / "dataset/ESPADA_depth_npy/ESPADA_PH/VAL")
	
	# train_loader, test_loader = prepare_data(rgb_dir, depth_dir, batch_size=4)
	

	dir = (root_dir / "dataset/nyudepthv2/train", 
		   root_dir / "dataset/nyudepthv2/val")
	train_loader, test_loader = prepare_data_tensorflow(dir, batch_size=8)

	
	# rgb_dir = (root_dir / "dataset/nyu_data/data/nyu2_train", 
	# 		   root_dir / "dataset/nyu_data/data/nyu2_test")
	# depth_dir = (root_dir / "dataset/nyu_data/data/nyu2_train", 
	# 			 root_dir / "dataset/nyu_data/data/nyu2_test")
	
	# train_loader, test_loader = prepare_data_kaggle(rgb_dir, depth_dir, batch_size=4)


	batch_counts = {}
	for i, data in enumerate(test_loader):
		images, depths = data['image'], data['depth']
		batch_count = batch_counts.get(len(images), 0) + 1
		batch_counts[len(images)] = batch_count
		print(f'Batch images shape: {images.shape}, Batch labels shape: {depths.shape}')
		print(f"Batch counts: {batch_counts}")

		# print(depths.shape, depths.dtype)
		# depths = 1000 / depths
		# depths = 1 / depths
		# depths = depths.permute(0, 2, 3, 1).cpu().numpy()   	
		# for i in range(len(depths)):
		# 	print(np.max(depths[i]), np.min(depths[i]))
		# 	print(depths[i].shape, np.max(depths[i]), np.min(depths[i]), depths[i].dtype)
		# 	plt.imshow(depths[i], cmap='rainbow', vmax=1, vmin=0)
		# 	plt.title('Colored Tensor Image')
		# 	#plt.axis('off')
		# 	plt.show()


	batch_counts = {}
	depth_max, depth_min = 0, 1000
	for i, data in enumerate(train_loader):
		images, depths = data['image'], data['depth']
		batch_count = batch_counts.get(len(images), 0) + 1
		batch_counts[len(images)] = batch_count
		#print(f'Batch images shape: {images.shape}, Batch labels shape: {depths.shape}')
		print(f"Batch counts: {batch_counts}", depth_max, depth_min)
		

