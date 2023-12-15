"""
THIS PROGRAM CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST
These classes will be called by training and test programmes
"""


import random
from torch.utils.data import Dataset
import PIL.ImageOps
from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision.datasets as dset

from config import PARAMETERS


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")

trainDir = os.path.join(baseDir, "DATASETS", "FRIBURGO", "Entrenamiento")
folderDataset = dset.ImageFolder(root=trainDir)
rooms = folderDataset.classes


def get_coords(imageDir):
    idxX = imageDir.index('_x')
    idxY = imageDir.index('_y')
    idxA = imageDir.index('_a')

    x = float(imageDir[idxX + 2:idxY])
    y = float(imageDir[idxY + 2:idxA])
    return x, y


class TrainCoarseLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        ancTuple = random.choice(self.imageFolderDataset.imgs)

        posTuple = random.choice(self.imageFolderDataset.imgs)
        while ancTuple[1] != posTuple[1] or ancTuple[0] == posTuple[0]:
            posTuple = random.choice(self.imageFolderDataset.imgs)

        negTuple = random.choice(self.imageFolderDataset.imgs)
        while ancTuple[1] == negTuple[1]:
            negTuple = random.choice(self.imageFolderDataset.imgs)

        imgAnc, imgPos, imgNeg = ancTuple[0], posTuple[0], negTuple[0]

        anchor, positive, negative = Image.open(imgAnc), Image.open(imgPos), Image.open(imgNeg)
        anchor, positive, negative = anchor.convert("RGB"), positive.convert("RGB"), negative.convert("RGB")

        if self.should_invert:
            anchor, positive, negative = \
                PIL.ImageOps.invert(anchor), PIL.ImageOps.invert(positive), PIL.ImageOps.invert(negative)

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return PARAMETERS.epochLengthCoarseLoc


class ValidationCoarseLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        valCSV = pd.read_csv(csvDir + '/ValidationCoarseLoc.csv')

        self.imgList = valCSV['Img Val']
        self.idxRoom = valCSV['Idx Room']
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img = self.imgList[index]
        idx = self.idxRoom[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idx

    def __len__(self):
        return len(self.imgList)


class TestCoarseLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=None, should_invert=True):

        testCSV = pd.read_csv(csvDir + '/Test' + illumination + 'CoarseLoc.csv')

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

        self.imgTestList = testCSV['Img Test']
        self.idxRoom = testCSV['Idx Room']

    def __getitem__(self, index):

        img, idxRoom = self.imgTestList[index], self.idxRoom[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom

    def __len__(self):
        return len(self.imgTestList)


class RepresentativeImages(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        testCSV = pd.read_csv(csvDir + '/RepresentativeImages.csv')

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert
        self.imgRepList, self.idxRoom = testCSV['Image'], testCSV['Idx Room']

    def __getitem__(self, index):

        imageRep = self.imgRepList[index]
        idxRoom = self.idxRoom[index]

        imageRep = Image.open(imageRep)
        imageRep = imageRep.convert("RGB")

        if self.should_invert:
            imageRep = PIL.ImageOps.invert(imageRep)

        if self.transform is not None:
            imageRep = self.transform(imageRep)

        return imageRep, idxRoom

    def __len__(self):
        return len(self.imgRepList)


class TrainFineLoc(Dataset):

    # def __init__(self, currentRoom, imageFolderDataset, transform=None, should_invert=True):
    def __init__(self, rNeg, currentRoom, imageFolderDataset, transform=None, should_invert=True):

        self.room = currentRoom
        self.rNeg = rNeg

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        ancTuple = random.choice(self.imageFolderDataset.imgs)
        while rooms[ancTuple[1]] != self.room:
            ancTuple = random.choice(self.imageFolderDataset.imgs)

        ancX, ancY = get_coords(ancTuple[0])
        coordsImgAnc = np.array([ancX, ancY])
        imgAnc = ancTuple[0]

        posTuple = random.choice(self.imageFolderDataset.imgs)
        posX, posY = get_coords(posTuple[0])
        coordsImgPos = np.array([posX, posY])
        dist = np.linalg.norm(coordsImgAnc - coordsImgPos)
        while posTuple[0] == ancTuple[0] or posTuple[1] != ancTuple[1] or dist > PARAMETERS.rPosFineLoc:
            posTuple = random.choice(self.imageFolderDataset.imgs)
            posX, posY = get_coords(posTuple[0])
            coordsImgPos = np.array([posX, posY])
            dist = np.linalg.norm(coordsImgAnc - coordsImgPos)
        imgPos = posTuple[0]

        negTuple = random.choice(self.imageFolderDataset.imgs)
        negX, negY = get_coords(negTuple[0])
        coordsImgNeg = np.array([negX, negY])
        dist = np.linalg.norm(coordsImgAnc - coordsImgNeg)
        # while negTuple[1] != ancTuple[1] or dist < PARAMETERS.rNegFineLoc:
        while negTuple[1] != ancTuple[1] or dist < self.rNeg:
            negTuple = random.choice(self.imageFolderDataset.imgs)
            negX, negY = get_coords(negTuple[0])
            coordsImgNeg = np.array([negX, negY])
            dist = np.linalg.norm(coordsImgAnc - coordsImgNeg)
        imgNeg = negTuple[0]

        anchor, positive, negative = Image.open(imgAnc), Image.open(imgPos), Image.open(imgNeg)
        anchor, positive, negative = anchor.convert("RGB"), positive.convert("RGB"), negative.convert("RGB")

        if self.should_invert:
            anchor, positive, negative = \
                PIL.ImageOps.invert(anchor), PIL.ImageOps.invert(positive), PIL.ImageOps.invert(negative)

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return PARAMETERS.epochLengthFineLoc


class ValidationFineLoc(Dataset):

    def __init__(self, currentRoom, imageFolderDataset, transform=None, should_invert=True):

        valCSV = pd.read_csv(csvDir + '/ValidationFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordsX, self.coordsY = \
            valCSV['Img'], valCSV['Coord X'], valCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TestFineLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=None, should_invert=True):

        testCSV = pd.read_csv(csvDir + "/Test" + illumination + "FineLoc.csv")

        self.imgsDir = testCSV['Img Test']
        self.idxRoom = testCSV['Idx Room']
        self.coordsX = testCSV['Coord X']
        self.coordsY = testCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img = self.imgsDir[index]
        idxRoom = self.idxRoom[index]
        coordX = self.coordsX[index]
        coordY = self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class VisualModelTrainFineLoc(Dataset):

    def __init__(self, currentRoom, imageFolderDataset, transform=None, should_invert=True):

        visualModelCSV = pd.read_csv(csvDir + '/TrainFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordsX, self.coordsY = \
            visualModelCSV['Img'], visualModelCSV['Coord X'], visualModelCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class VisualModelTestFineLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        visualModelCSV = pd.read_csv(csvDir + '/VisualModelFineLoc.csv')

        self.imgsDir = visualModelCSV['Img']
        self.idxRoom = visualModelCSV['Idx Room']
        self.coordsX = visualModelCSV['Coord X']
        self.coordsY = visualModelCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img = self.imgsDir[index]
        idxRoom = self.idxRoom[index]
        coordX = self.coordsX[index]
        coordY = self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class VisualModelGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        visualModelCSV = pd.read_csv(csvDir + '/VisualModelGlobalLoc.csv')

        self.imgsDir = visualModelCSV['Img']
        self.coordsX = visualModelCSV['Coord X']
        self.coordsY = visualModelCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img = self.imgsDir[index]
        coordX = self.coordsX[index]
        coordY = self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TrainGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        ancTuple = random.choice(self.imageFolderDataset.imgs)
        ancX, ancY = get_coords(ancTuple[0])
        coordsImgAnc = np.array([ancX, ancY])
        imgAnc = ancTuple[0]

        posTuple = random.choice(self.imageFolderDataset.imgs)
        posX, posY = get_coords(posTuple[0])
        coordsImgPos = np.array([posX, posY])
        dist = np.linalg.norm(coordsImgAnc - coordsImgPos)
        while posTuple[0] == ancTuple[0] or dist > PARAMETERS.rPosFineLoc:
            posTuple = random.choice(self.imageFolderDataset.imgs)
            posX, posY = get_coords(posTuple[0])
            coordsImgPos = np.array([posX, posY])
            dist = np.linalg.norm(coordsImgAnc - coordsImgPos)
        imgPos = posTuple[0]

        negTuple = random.choice(self.imageFolderDataset.imgs)
        negX, negY = get_coords(negTuple[0])
        coordsImgNeg = np.array([negX, negY])
        dist = np.linalg.norm(coordsImgAnc - coordsImgNeg)
        while dist < PARAMETERS.rNegFineLoc:
            negTuple = random.choice(self.imageFolderDataset.imgs)
            negX, negY = get_coords(negTuple[0])
            coordsImgNeg = np.array([negX, negY])
            dist = np.linalg.norm(coordsImgAnc - coordsImgNeg)
        imgNeg = negTuple[0]

        anchor, positive, negative = Image.open(imgAnc), Image.open(imgPos), Image.open(imgNeg)
        anchor, positive, negative = anchor.convert("RGB"), positive.convert("RGB"), negative.convert("RGB")

        if self.should_invert:
            anchor, positive, negative = \
                PIL.ImageOps.invert(anchor), PIL.ImageOps.invert(positive), PIL.ImageOps.invert(negative)

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return PARAMETERS.epochLengthGlobalLoc


class ValidationGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        valCSV = pd.read_csv(csvDir + '/ValidationGlobalLoc.csv')

        self.imgsDir, self.coordsX, self.coordsY = \
            valCSV['Img'], valCSV['Coord X'], valCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TestGlobalLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=None, should_invert=True):

        testCSV = pd.read_csv(csvDir + "/Test" + illumination + "GlobalLoc.csv")

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

        self.imgsDir = testCSV['Img']
        self.coordsX = testCSV['Coord X']
        self.coordsY = testCSV['Coord Y']

    def __getitem__(self, index):

        img = self.imgsDir[index]
        coordX = self.coordsX[index]
        coordY = self.coordsY[index]

        img = Image.open(img)
        img = img.convert("RGB")

        if self.should_invert:
            img = PIL.ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)
