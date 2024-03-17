from pathlib import Path
from torchvision.datasets.utils import download_and_extract_archive
# Download URL for the Dataset
url = "https://storage.googleapis.com/kaggle-data-sets/4046394/7034146/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240118%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240118T141037Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=3e9e70633bdd572747aa6dd9b0019ae93aec6db28d77cd499272bf6bd77aff6a7fbbc6f81cea2b46178c1531fedd9f8c815d08ab5669523b93ca78b030fb519f5ae3bbc89f61a283e6ae4771f2c89f9c331df701c7f13e2443de2df58808337674cc839b34599f40f82e0c31ffea56dc4279ef8911a6c2245faf48949ad09478c6dd8867f9cec4eccfca8c3af6dc48c3054dd968bd6ce82ae621c12b56e71f34ebc4ea65003ac7effbfad5896d56b835120b6a650b3cca53df01ceb3dc042ba744f625cfbf790254f4ea38f68929e8697a970fcf62821807bf8fca61358adbb0651044aac8e304f516a731fb55d859bd4025feb542332e1d8e6f31698e684101"

root = Path(__file__).parent
data_dir = root / Path("dataset")

if not Path("archive.zip").exists():
    download_and_extract_archive(url, 
                                 download_root = root, 
                                 extract_root = data_dir, 
                                 filename = "archive.zip")
    print("Download completed")