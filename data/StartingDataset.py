import torch
<<<<<<< HEAD
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
=======
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor

>>>>>>> dac93683bea42deeb15d8f98abea782feeb8ec40

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
<<<<<<< HEAD
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
=======

    def __init__(self, csv_path, train=True):
        if train:
            relative_path = "train_images/"
        else:
            relative_path = "test_images/"

        df = pd.read_csv(csv_path)
        self.image_ids = list(df['image_id'])
        for id,i in enumerate(self.image_ids):
            self.image_ids[i] = relative_path + id
        self.labels = list(df["label"])
        self.l = len(self.labels)

    def __getitem__(self, index):
        #inputs = torch.zeros([3, 224, 224])
        image = Image(self.image_ids[index])
        inputs = ToTensor(image)
>>>>>>> dac93683bea42deeb15d8f98abea782feeb8ec40

        return inputs, self.labels[index]

    def __len__(self): #Assumes size of class doesn't change
        return self.l 
