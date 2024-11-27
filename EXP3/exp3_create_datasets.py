"""
THIS SCRIPT CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST IN EXP3
The classes will be called by training and test scripts
The classes read the CSV files with the image paths, convert the images into tensors and load them into the CPU/GPU
"""


from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from config import PARAMS
from functions import process_image
import os


csvDir = os.path.join(PARAMS.csvDir, "EXP3")


class TrainCoarseLoc(Dataset):

    def __init__(self, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/TrainCoarseLoc.csv')

        self.transform = transform
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV['ImgAnc'], CSV['ImgPos'], CSV['ImgNeg']

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index],

        anchor = process_image(imgAnc, self.transform)
        positive = process_image(imgPos, self.transform)
        negative = process_image(imgNeg, self.transform)

        return anchor, positive, negative

    def __len__(self):
        return len(self.imgsAnc)



class TrainFineLoc(Dataset):

    def __init__(self, currentRoom, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/TrainFineLoc' + currentRoom + '.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV['ImgAnc'], CSV['ImgPos'], CSV['ImgNeg']
        self.transform = transform

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]

        anchor = process_image(imgAnc, self.transform)
        positive = process_image(imgPos, self.transform)
        negative = process_image(imgNeg, self.transform)

        return anchor, positive, negative

    def __len__(self):
        return len(self.imgsAnc)


class Validation(Dataset):

    def __init__(self, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/Validation.csv')

        self.imgList, self.idxRoom, self.coordX, self.coordY = \
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']
        self.transform = transform

    def __getitem__(self, index):
        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)


class Test(Dataset):

    def __init__(self, illumination, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/Test' + illumination + '.csv')

        self.transform = transform
        self.imgList, self.idxRoom, self.coordX, self.coordY =\
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']

    def __getitem__(self, index):

        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)


class RepImages(Dataset):

    def __init__(self, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/RepImages.csv')

        self.transform = transform
        self.imgList, self.idxRoom, self.coordX, self.coordY = \
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']

    def __getitem__(self, index):
        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)


class VisualModel(Dataset):

    def __init__(self, transform=transforms.ToTensor()):

        CSV = pd.read_csv(csvDir + '/VisualModel.csv')

        self.transform = transform
        self.imgList, self.idxRoom, self.coordX, self.coordY = \
            CSV['Img'], CSV['Idx Room'], CSV['Coord X'], CSV['Coord Y']

    def __getitem__(self, index):
        img, idxRoom, coordX, coordY = self.imgList[index], self.idxRoom[index], self.coordX[index], self.coordY[index]
        img = process_image(img, self.transform)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgList)
