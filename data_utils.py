import requests
import tarfile
import zipfile
import os

def download_and_extract_dataset(url, extract_to='data'):
	"""
	Downloads a dataset from the given URL and extracts it into the specified directory.

	Args:
		url (str): The URL where the dataset TAR.GZ file is located.
		extract_to (str): The directory path where the dataset will be extracted. Defaults to 'data'.

	Returns:
		None

	Raises:
		requests.exceptions.HTTPError: If the download from the URL fails or returns a non-200 status code.
		tarfile.TarError: If there is an error in extracting the tar file.
		IOError: If there is an error writing to the file system.

	This function first ensures that the directory specified by `extract_to` exists, creating it if necessary.
	It then downloads the file to a temporary tar.gz file within the target directory. If the download is successful,
	it proceeds to extract the contents of the tar.gz file to the specified directory. After extraction, the
	temporary tar.gz file is deleted.
	"""
	# Ensure the target directory exists
	os.makedirs(extract_to, exist_ok=True)

	# Define the name of the temp tar file
	temp_tar_path = os.path.join(extract_to, 'temp_dataset.tar.gz')

	# Download the dataset
	print("Downloading dataset...")
	response = requests.get(url, stream=True)
	try:
		response.raise_for_status()  # Raises an HTTPError for bad responses
		with open(temp_tar_path, 'wb') as f:
			for chunk in response.iter_content(chunk_size=8192):
				f.write(chunk)
		print("Download complete.")
	except requests.exceptions.HTTPError as e:
		print(f"Failed to download the file: {str(e)}")
		return

	# Extract the dataset
	try:
		print("Extracting dataset...")
		with tarfile.open(temp_tar_path, "r:gz") as tar:
			tar.extractall(path=extract_to)
		print("Extraction complete.")
	except tarfile.TarError as e:
		print(f"Failed to extract the tar file: {str(e)}")
		return

	# Optionally remove the temp tar file after extraction
	os.remove(temp_tar_path)
	print("Temporary file removed.")


def download_and_extract_zip_from_google_drive(google_drive_url, extract_to='.'):
	"""
	Downloads a ZIP file from Google Drive and extracts it to a specified directory.

	Args:
	google_drive_url (str): The URL of the Google Drive file to download. It should be a shareable link.
	extract_to (str): The path to the directory where the ZIP file should be extracted. Defaults to the current directory.

	Returns:
	None

	Raises:
	requests.exceptions.RequestException: If there is an issue with network access or with Google Drive's response.
	IOError: If there are issues reading or writing files during the process.
	"""
	# Convert the Google Drive share link to a download link
	file_id = google_drive_url.split('/d/')[1].split('/')[0]
	download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
	
	# Start the download process
	print("Downloading file from Google Drive...")
	session = requests.Session()
	response = session.get(download_url, stream=True)
	
	token = get_confirm_token(response)
	if token:
		params = {'confirm': token}
		response = session.get(download_url, params=params, stream=True)

	# Ensure the target directory exists
	os.makedirs(extract_to, exist_ok=True)

	# Define path for temporary download
	temp_zip_path = os.path.join(extract_to, 'temp_file.zip')

	# Write the content to a temporary file
	with open(temp_zip_path, 'wb') as f:
		for chunk in response.iter_content(chunk_size=8192):
			f.write(chunk)

	# Unzip the file into the directory
	print("Extracting file...")
	with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
		zip_ref.extractall(extract_to)
	print("Extraction complete.")
	
	# Remove the temporary zip file
	os.remove(temp_zip_path)
	print("Temporary file removed.")


def get_confirm_token(response):
	"""Helper function to retrieve the Google Drive download warning token."""
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None


def compress_experiment_data(source_dir, output_filename):
	"""
	Compresses the contents of the source directory into a zip file and saves it to the root directory.

	Args:
	source_dir (str): The path to the directory containing the data to compress.
	output_filename (str): The name of the output zip file to be created in the root directory.

	Returns:
	None
	"""
	# Create a full path for the output file in the root directory
	root_dir = './'
	output_path = os.path.join(root_dir, output_filename)

	# Creating a zip file
	with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
		# Walk through the directory
		for root, dirs, files in os.walk(source_dir):
			for file in files:
				# Create the full path to the file
				file_path = os.path.join(root, file)
				# Add file to the zip file
				# The arcname parameter avoids storing the full path in the zip file
				zipf.write(file_path, arcname=os.path.relpath(file_path, start=source_dir))
	
	print(f"Created zip file {output_filename} at {output_path}")


def main():
	# Usage
	google_drive_url = 'https://drive.google.com/file/d/1jg-qLjRoHOx8_cX2ziZXGCWGL5xhC25Q/view?usp=sharing'
	download_and_extract_zip_from_google_drive(google_drive_url, extract_to='.')

	dataset_url = 'http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz'
	download_and_extract_dataset(dataset_url, extract_to='DenseDepth/dataset')	
	
	compress_experiment_data('./runs', 'experiment_data.zip')
	compress_experiment_data('./checkpoint', 'experiment_checkpoint.zip')


if __name__ == "__main__":
	main()
