import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    paths = []
    
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.image_ids = list(data[image_id])
        self.labels = list(data[label])
        

    def __getitem__(self, index):
        inputs = torch.zeros([3, 224, 224])
        label = 0
        #get a single image

        #load images with Pillow
        #image = Image.open()

        return inputs, label

    def __len__(self):
        return 10000
