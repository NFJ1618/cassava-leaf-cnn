#import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
import torch
import matplotlib.pyplot as plt

def test(): 
    """
    Function to test StartingDataset tensors and image display
    """
    dataset = StartingDataset(csv_path='data/sample_submission_2.csv', folder_path='data/train_images', img_size=IMG_SIZE)
    for i in range(len(dataset)):
        image, label = dataset[i]
        plt.imshow(image.permute(1,2,0)) #3rd RBG dimension being first confuses imshow
        print('Label: ', label)
        plt.show(block=True) #Allows image to shown even when main called from terminal


def run(kaggle_path=""):
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)
    print("Kaggle path:", kaggle_path)

    # Initalize dataset and model. Then train the model!
    if not kaggle_path:
        train_dataset = StartingDataset(csv_path='data/train.csv', folder_path='data/train_images', img_size=IMG_SIZE)
        val_dataset = StartingDataset(csv_path='data/train.csv', folder_path='data/train_images', img_size=IMG_SIZE)
    else:
    #For Kaggle
        train_dataset = StartingDataset(csv_path=kaggle_path + '/train.csv', folder_path=kaggle_path + '/train_images', img_size=IMG_SIZE)
        val_dataset = StartingDataset(csv_path=kaggle_path + '/train.csv', folder_path=kaggle_path + '/train_images', img_size=IMG_SIZE)
    print("Data in class")
    model = StartingNetwork()
    model = model.to(device)
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device = device
    )


if __name__ == "__main__":
    #test()
    run()
