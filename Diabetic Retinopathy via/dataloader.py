import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import DatasetFolder

def img_loader(path):
	"""
	Custom loader to load an image from the given path and convert it to 'RGB'.

	Args:
	- path (str): The file path to the image.

	Returns:
	- Image: The loaded and converted image.
	"""
	return Image.open(path).convert('RGB')


def initialize_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=8, transform=None):
	"""
	Initializes and returns a PyTorch DataLoader along with the dataset.
	The target size to which the images will be resized. Defaults to 224.

	Args:
	- data_dir (str): The directory where the dataset is stored.
	- batch_size (int, optional): The batch size for the DataLoader. Defaults to 32.
	- shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
	- num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 8.
	- transform (torchvision.transforms.Compose, optional): The transformation to apply to the dataset images. Defaults to None.

	Returns:
	- DataLoader: The initialized DataLoader.
	- DatasetFolder: The dataset used in the DataLoader.
	"""
	if transform == None:
		transform = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
		])
		
	# Initialize dataset with the provided transform (if any)
	dataset = DatasetFolder(root=data_dir, 
							loader=img_loader, 
							extensions=('jpg', 'png'),
							transform=transform)

	# Initialize DataLoader
	dataloader = torch.utils.data.DataLoader(dataset, 
											 batch_size=batch_size, 
											 shuffle=shuffle, 
											 num_workers=num_workers)
	return dataloader, dataset


# Load and Preprocess Dataset
def prepare_data(train_data_dir, valid_data_dir, batch_size, target_size):
	trai_transform = transforms.Compose([
		transforms.RandomHorizontalFlip(p=1.0),
		transforms.RandomVerticalFlip(p=1.0),
		transforms.RandomRotation((-30, 30)),
		transforms.Resize((target_size, target_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	val_transform = transforms.Compose([
		transforms.Resize((target_size, target_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	train_dataloader, _ = initialize_dataloader(
		train_data_dir, 
		batch_size=batch_size, 
		transform=trai_transform)

	valid_dataloader, _ = initialize_dataloader(
		valid_data_dir, 
		batch_size=batch_size, 
		transform=val_transform,
		shuffle=False)
	return train_dataloader, valid_dataloader


if __name__ == '__main__':
	root_dir = Path(__file__).parent
	train_data_dir = root_dir / "dataset" / "train"
	valid_data_dir = root_dir / "dataset" / "valid"
	target_size=224
	batch_size=32
	
	train_dataloader, train_dataset = initialize_dataloader(train_data_dir, batch_size)
	print(train_dataset)
	print(f"Batch count: {len(train_dataloader)}")


	# Data Loader
	train_loader, test_loader = prepare_data(train_data_dir, valid_data_dir, batch_size, target_size)
	
	for batch_idx, (images, labels) in enumerate(train_loader):
		print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
		print(labels)
		break

	for batch_idx, (images, labels) in enumerate(test_loader):
		print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
		print(labels)
		break

"""
Dataset DatasetFolder
	Number of datapoints: 2076
	Root location: C:\\Users\\AAA\\Documents\\repos\\Python-Project\\Python-Project\\Diabetic Retinopathy via\\dataset\\train
	StandardTransform
Transform: Compose(
				Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
				ToTensor()
			)
Batch count: 65

Batch 0: Images shape: torch.Size([32, 3, 224, 224]), Labels shape: torch.Size([32])
tensor([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
		0, 0, 1, 1, 0, 0, 1, 1])

Batch 0: Images shape: torch.Size([32, 3, 224, 224]), Labels shape: torch.Size([32])
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0])
"""