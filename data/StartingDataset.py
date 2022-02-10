from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor, functional

class StartingDataset(Dataset):
    """
    Load on demand style dataset that will also do image transformations for required dimensions each time
    """
    def __init__(self, csv_path, folder_path, img_size):
        df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.image_ids = list(df['image_id'])
        for i,id in enumerate(self.image_ids):
            self.image_ids[i] = folder_path + '/' + id
        self.labels = list(df["label"])
        self.l = len(self.labels)

    def __getitem__(self, index):
        #inputs = torch.zeros([3, 224, 224])
        inputs = 0
        with Image.open(self.image_ids[index]) as image:
            inputs = ToTensor()(image)
            inputs = functional.resize(inputs, self.img_size)

        return inputs, self.labels[index]

    def __len__(self): #Assumes size of class doesn't change
        return self.l 
