from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor, functional
from random import randint
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class StartingDataset(Dataset):
    """
    Load on demand style dataset that will also do image transformations for required dimensions each time
    """
    def __init__(self, csv_path, folder_path, img_size, transform=None, data_ratio=1.0):
        df = pd.read_csv(csv_path)
        
        # added for quick test runs.
        df = df[:int(data_ratio * df.shape[0])]

        self.img_size = img_size
        self.image_ids = list(df['image_id'])
        for i,id in enumerate(self.image_ids):
            self.image_ids[i] = folder_path + '/' + id
        self.labels = list(df["label"])
        self.l = len(self.labels)
        self.transform = transform

    def __getitem__(self, index):
        #inputs = torch.zeros([3, 224, 224])
        inputs = 0
        with Image.open(self.image_ids[index]) as image:
            inputs = functional.resize(image, self.img_size)

        inputs = ToTensor()(inputs)
        if self.transform:
            inputs = self.transform(inputs)
        
        return inputs, self.labels[index]

    def __len__(self): #Assumes size of class doesn't change
        return self.l

    def split_test_train_data(self, dataset, batch_size, ratio=0.2):
        SEED = randint(0,100)

        # generate indices: instead of the actual data we pass in integers instead
        train_indices, test_indices, _, _ = train_test_split(
            range(self.l),
            self.labels,
            stratify=self.labels,
            test_size=ratio,
            random_state=SEED
        )

        # generate subset based on indices
        train_split = Subset(dataset, train_indices)
        test_split = Subset(dataset, test_indices)

        # create batches
        train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(test_split, batch_size=batch_size, shuffle=True)

        return train_loader, validation_loader