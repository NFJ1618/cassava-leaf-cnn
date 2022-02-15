import constants
import numpy as np
import cv2
import statistics 

from data.StartingDataset import StartingDataset
dataset = StartingDataset(csv_path='data/train.csv', folder_path='data/train_images', img_size=constants.IMG_SIZE)

blue = []
green = []
red = []
for i,id in enumerate(self.image_ids):
    myimg = cv2.imread(self.image_ids[i])
    avg_color_per_row = np.average(myimg, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)/256
    blue.append(avg_color[0])
    green.append(avg_color[1])
    red.append(avg_color[2])

print('Blue Mean', sum(blue)/len(blue))
print('Blue std', statistics.pstdev(blue))

print('Green Mean', sum(green)/len(green))
print('Green std', statistics.pstdev(green))

print('Red Mean', sum(red)/len(red))
print('Red std', statistics.pstdev(red))