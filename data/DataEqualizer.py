import pandas as pd
from PIL import Image
from torchvision import transforms

df = pd.read_csv('train.csv')
targetNum = 13158


transform_arr = [transforms.functional.hflip(), transforms.functional.vflip(), transforms.functional.rotate(angle=45)]

for label in range(5):
    num = df['label'].value_counts()[label]
    label_set = df['label' == label]
    i = num
    while i < targetNum:
        j = 0
        while i < num and i < targetNum:
            image_id = label_set.loc[i, 'image_id']
            image_path = 'train_images/' + image_id
            with Image.open(image_path) as image:
                im_tensor = transforms.ToTensor()(image)
            new_im_tensor = transform_arr[j](im_tensor)
            new_image = transforms.ToPILImage()(new_im_tensor)
            new_image.save('train_images/image_id' + '_' + j)
            i += 1
        j += 1
        label_set = df['label' == label]
        num = len(label_set)


