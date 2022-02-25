import torch
from PIL import Image
import torchvision.transforms as transforms
import constants
from math import sqrt

from data.StartingDataset import StartingDataset
dataset = StartingDataset(csv_path='data/train.csv', folder_path='data/train_images', img_size=constants.IMG_SIZE[0])

transform = transforms.ToTensor()

# Read the input image
imgTensor = torch.zeros(3,224,224)
#temp = torch.zeros(3,224,224)
for i,id in enumerate(dataset.image_ids):
    with Image.open(id) as image:
        image = transforms.functional.resize(image, constants.IMG_SIZE[0])
        imgTensor += transform(image)

# Compute mean of the Image Tensor across image channels RGB
R_mean, G_mean ,B_mean = torch.mean(imgTensor, dim = [1,2])

# print mean across image channel RGB
print("Mean across Read channel:", R_mean/len(dataset))
print("Mean across Green channel:", G_mean/len(dataset))
print("Mean across Blue channel:", B_mean/len(dataset))

R_dev = G_dev = B_dev = 0

for i,id in enumerate(dataset.image_ids):
    with Image.open(id) as image:
        image = transforms.functional.resize(image, constants.IMG_SIZE[0])
        image = transform(image)
        R, G, B = torch.mean(image, dim = [1,2])
        R_dev, G_dev, B_dev = R_dev+abs(R-R_mean/len(dataset))**2, G_dev+abs(G-G_mean/len(dataset))**2, B_dev+abs(B-B_mean/len(dataset))**2

print("Dev across Read channel:", sqrt(R_dev/len(dataset)))
print("Dev across Green channel:", sqrt(G_dev/len(dataset)))
print("Dev across Blue channel:", sqrt(B_dev/len(dataset)))