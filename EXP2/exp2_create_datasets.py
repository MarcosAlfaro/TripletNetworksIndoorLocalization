"""
THIS SCRIPT CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO THE TEST IN EXP2
The classes will be called by training and test scripts
The classes read the CSV files with the image paths, convert the images into tensors and load them into the CPU/GPU
"""
import os.path

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from config import PARAMS
from functions import process_image

csvDir = os.path.join(PARAMS.csvDir, "EXP2")


class Test(Dataset):

    def __init__(self, illumination, effect, value, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/' + effect + '/Test' + illumination + str(value) + '.csv')

        self.imgList, self.idxRoom, self.coordX, self.coordY =\
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']
        self.transform = transform

    def __getitem__(self, index):

        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)


class VisualModel(Dataset):

    def __init__(self, effect, value, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/' + effect + '/VisualModel' + str(value) + '.csv')

        self.transform = transform
        self.imgList, self.idxRoom, self.coordX, self.coordY = \
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']

    def __getitem__(self, index):
        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)


class RepImages(Dataset):

    def __init__(self, effect, value, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/' + effect + '/RepImgs' + str(value) + '.csv')

        self.transform = transform
        self.imgList, self.idxRoom, self.coordX, self.coordY = \
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']

    def __getitem__(self, index):
        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)
