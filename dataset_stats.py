import torch
from PIL import Image
import torchvision.transforms as transforms
import constants

from data.StartingDataset import StartingDataset
dataset = StartingDataset(csv_path='data/train.csv', folder_path='data/train_images', img_size=constants.IMG_SIZE[0])

transform = transforms.ToTensor()
# Read the input image
imgTensor = torch.zeros(3,224,224)
#temp = torch.zeros(3,224,224)
c = 0
for i,id in enumerate(dataset):
    imgTensor += id[0]
    c += 1
    if c % 1000 == 0:
        print(torch.mean(imgTensor, dim = [1,2]))

# Compute mean of the Image Tensor across image channels RGB
R_mean, G_mean ,B_mean = torch.mean(imgTensor, dim = [1,2])

# print mean across image channel RGB
print("Mean across Read channel:", R_mean/(21397))
print("Mean across Green channel:", G_mean/(21397))
print("Mean across Blue channel:", B_mean/(21397))