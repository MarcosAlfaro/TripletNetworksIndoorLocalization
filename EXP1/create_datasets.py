"""
THIS PROGRAM CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST
These classes will be called by training and test programmes
"""


import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dset
from PIL import Image, ImageOps
import pandas as pd
import os
import numpy as np
from config import PARAMETERS


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")

trainDir = os.path.join(baseDir, "DATASETS", "FRIBURGO", "Train")
folderDataset = dset.ImageFolder(root=trainDir)
rooms = folderDataset.classes


def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return [x, y]


class TrainCoarseLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

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
                ImageOps.invert(anchor), ImageOps.invert(positive), ImageOps.invert(negative)

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return PARAMETERS.epochLengthCoarseLoc


class ValidationCoarseLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        valCSV = pd.read_csv(csvDir + '/Exp1ValidationCoarseLoc.csv')

        self.imgList, self.idxRoom = valCSV['Img Val'], valCSV['Idx Room']
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxRoom = self.imgList[index], self.idxRoom[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom

    def __len__(self):
        return len(self.imgList)


class TestCoarseLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + '/Exp1Test' + illumination + 'CoarseLoc.csv')

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert
        self.imgTestList, self.idxRoom = testCSV['Img Test'], testCSV['Idx Room']

    def __getitem__(self, index):

        img, idxRoom = self.imgTestList[index], self.idxRoom[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom

    def __len__(self):
        return len(self.imgTestList)


class RepresentativeImages(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + '/Exp1RepresentativeImages.csv')

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert
        self.imgRepList, self.idxRoom = testCSV['Image'], testCSV['Idx Room']

    def __getitem__(self, index):

        imageRep, idxRoom = self.imgRepList[index], self.idxRoom[index]
        imageRep = Image.open(imageRep).convert("RGB")

        if self.should_invert:
            imageRep = ImageOps.invert(imageRep)

        if self.transform is not None:
            imageRep = self.transform(imageRep)

        return imageRep, idxRoom

    def __len__(self):
        return len(self.imgRepList)


class TrainFineLoc(Dataset):

    def __init__(self, tree, currentRoom, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        trainCSV = pd.read_csv(csvDir + '/Exp1TrainFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordX, self.coordY = trainCSV['Img'], trainCSV['Coord X'], trainCSV['Coord Y']

        self.tree, self.room = tree, currentRoom
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        rPos = PARAMETERS.rPosFineLoc
        rNeg = PARAMETERS.rNegFineLoc

        idxAnc = index % len(self.imgsDir)
        imgAnc = self.imgsDir[idxAnc]
        coordsImgAnc = np.array([self.coordX[idxAnc], self.coordY[idxAnc]])

        posIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rPos)[0].reshape(1, -1)[0]
        d = 0
        while len(posIdxs) <= 1:
            d += 0.01
            posIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rPos + d)[0].reshape(1, -1)[0]

        idxPos = np.random.choice(posIdxs)
        while idxPos == idxAnc:
            idxPos = np.random.choice(posIdxs)
        imgPos = self.imgsDir[idxPos]

        negIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rNeg)[0].reshape(1, -1)[0]

        idxNeg = np.random.choice(range(0, len(self.imgsDir)))
        while idxNeg in negIdxs:
            idxNeg = np.random.choice(range(0, len(self.imgsDir)))
        imgNeg = self.imgsDir[idxNeg]

        anchor, positive, negative = Image.open(imgAnc), Image.open(imgPos), Image.open(imgNeg)
        anchor, positive, negative = anchor.convert("RGB"), positive.convert("RGB"), negative.convert("RGB")

        if self.should_invert:
            anchor, positive, negative = \
                ImageOps.invert(anchor), ImageOps.invert(positive), ImageOps.invert(negative)

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return PARAMETERS.epochLengthFineLoc


class ValidationFineLoc(Dataset):

    def __init__(self, currentRoom, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        valCSV = pd.read_csv(csvDir + '/Exp1ValidationFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordsX, self.coordsY = \
            valCSV['Img'], valCSV['Coord X'], valCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TestFineLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + "/Exp1Test" + illumination + "FineLoc.csv")

        self.imgsDir, self.idxRoom, self.coordsX, self.coordsY \
            = testCSV['Img Test'], testCSV['Idx Room'], testCSV['Coord X'], testCSV['Coord Y']
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxRoom, coordX, coordY \
            = self.imgsDir[index], self.idxRoom[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class VisualModelTrainFineLoc(Dataset):

    def __init__(self, currentRoom, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        vmCSV = pd.read_csv(csvDir + '/Exp1TrainFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordsX, self.coordsY = vmCSV['Img'], vmCSV['Coord X'], vmCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class VisualModelTestFineLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        vmCSV = pd.read_csv(csvDir + '/Exp1VisualModelFineLoc.csv')

        self.imgsDir, self.idxRoom, self.coordsX, self.coordsY \
            = vmCSV['Img'], vmCSV['Idx Room'], vmCSV['Coord X'], vmCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxRoom, coordX, coordY \
            = self.imgsDir[index], self.idxRoom[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class VisualModelGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        vmCSV = pd.read_csv(csvDir + '/Exp1VisualModelGlobalLoc.csv')

        self.imgsDir, self.coordsX, self.coordsY = vmCSV['Img'], vmCSV['Coord X'], vmCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TrainGlobalLoc(Dataset):

    def __init__(self, tree, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        trainCSV = pd.read_csv(csvDir + '/Exp1TrainGlobalLoc.csv')

        self.imgsDir, self.coordX, self.coordY = trainCSV['Img'], trainCSV['Coord X'], trainCSV['Coord Y']

        self.tree = tree
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        rPos = PARAMETERS.rPosGlobalLoc
        rNeg = PARAMETERS.rNegGlobalLoc

        idxAnc = index % len(self.imgsDir)
        imgAnc = self.imgsDir[idxAnc]
        coordsImgAnc = np.array([self.coordX[idxAnc], self.coordY[idxAnc]])

        posIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rPos)[0].reshape(1, -1)[0]
        d = 0
        while len(posIdxs) <= 1:
            d += 0.01
            posIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rPos + d)[0].reshape(1, -1)[0]

        idxPos = np.random.choice(posIdxs)
        while idxPos == idxAnc:
            idxPos = np.random.choice(posIdxs)
        imgPos = self.imgsDir[idxPos]

        negIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rNeg)[0].reshape(1, -1)[0]

        idxNeg = np.random.choice(range(0, len(self.imgsDir)))
        while idxNeg in negIdxs:
            idxNeg = np.random.choice(range(0, len(self.imgsDir)))
        imgNeg = self.imgsDir[idxNeg]

        anchor, positive, negative = Image.open(imgAnc), Image.open(imgPos), Image.open(imgNeg)
        anchor, positive, negative = anchor.convert("RGB"), positive.convert("RGB"), negative.convert("RGB")

        if self.should_invert:
            anchor, positive, negative = \
                ImageOps.invert(anchor), ImageOps.invert(positive), ImageOps.invert(negative)

        if self.transform is not None:
            anchor, positive, negative = self.transform(anchor), self.transform(positive), self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return PARAMETERS.epochLengthFineLoc


class ValidationGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        valCSV = pd.read_csv(csvDir + '/Exp1ValidationGlobalLoc.csv')

        self.imgsDir, self.coordsX, self.coordsY = \
            valCSV['Img'], valCSV['Coord X'], valCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TestGlobalLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + "/Exp1Test" + illumination + "GlobalLoc.csv")

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

        self.imgsDir, self.coordsX, self.coordsY = testCSV['Img'], testCSV['Coord X'], testCSV['Coord Y']

    def __getitem__(self, index):

        img, coordX, coordY = self.imgsDir[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)
