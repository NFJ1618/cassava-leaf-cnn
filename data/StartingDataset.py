import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

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

        return inputs, self.labels[index]

    def __len__(self): #Assumes size of class doesn't change
        return self.l 
