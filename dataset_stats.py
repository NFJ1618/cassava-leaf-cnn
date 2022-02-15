import torch
import constants

from data.StartingDataset import StartingDataset
dataset = StartingDataset(csv_path='data/train.csv', folder_path='data/train_images', img_size=constants.IMG_SIZE)

for channel in range(3):
    channel_arr = torch.empty((0))
    for image in dataset:
        channel_arr = torch.cat([channel_arr, image[0][channel].flatten()])
    print(channel)
    print('Mean: ', torch.mean(channel_arr))
    print('Stdev: ', torch.std(channel_arr))




