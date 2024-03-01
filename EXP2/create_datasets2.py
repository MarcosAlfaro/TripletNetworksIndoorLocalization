"""
THIS PROGRAM CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST
These classes will be called by training and test programmes
"""


import random
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import pandas as pd
import os
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

from config import PARAMETERS


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")

trainDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS", "Train")
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

        valCSV = pd.read_csv(csvDir + '/Exp2ValidationCoarseLoc.csv')

        self.imgList, self.idxEnv, self.idxRoom = valCSV['Img Val'], valCSV['Idx Env'], valCSV['Idx Room']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxEnv, idxRoom = self.imgList[index], self.idxEnv[index], self.idxRoom[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, idxRoom

    def __len__(self):
        return len(self.imgList)


class TestCoarseLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + '/Exp2Test' + illumination + 'CoarseLoc.csv')

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

        self.imgTestList, self.idxEnv, self.idxRoom = testCSV['Img Test'], testCSV['Idx Env'], testCSV['Idx Room']

    def __getitem__(self, index):

        img, idxEnv, idxRoom = self.imgTestList[index], self.idxEnv[index], self.idxRoom[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, idxRoom

    def __len__(self):
        return len(self.imgTestList)


class RepresentativeImages(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + '/Exp2RepresentativeImages.csv')

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

        trainCSV = pd.read_csv(csvDir + '/Exp2TrainFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordX, self.coordY = trainCSV['Img'], trainCSV['Coord X'], trainCSV['Coord Y']

        self.tree, self.room = tree, currentRoom
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        rPos, rNeg = PARAMETERS.rPosFineLoc, PARAMETERS.rNegFineLoc

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

        idxNeg = np.random.choice(range(len(self.imgsDir)))
        while idxNeg in negIdxs:
            idxNeg = np.random.choice(range(len(self.imgsDir)))
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


class VisualModelTrainFineLoc(Dataset):

    def __init__(self, currentRoom, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        visualModelCSV = pd.read_csv(csvDir + '/Exp2VisualModelTrainFineLoc' + currentRoom + '.csv')

        self.imgsDir, self.coordsX, self.coordsY = \
            visualModelCSV['Img'], visualModelCSV['Coord X'], visualModelCSV['Coord Y']

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


class ValidationFineLoc(Dataset):

    def __init__(self, currentRoom, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        valCSV = pd.read_csv(csvDir + '/Exp2ValidationFineLoc' + currentRoom + '.csv')

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


class VisualModelTestFineLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        vmCSV = pd.read_csv(csvDir + '/Exp2VisualModelTestFineLoc' + '.csv')

        self.imgsDir,  self.idxRoom, self.idxEnv, self.coordsX, self.coordsY = \
            vmCSV['Img'], vmCSV['Idx Room'], vmCSV['Idx Env'], vmCSV['Coord X'], vmCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxEnv, idxRoom, coordX, coordY = \
            self.imgsDir[index], self.idxEnv[index], self.idxRoom[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TestFineLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + "/Exp2Test" + illumination + "FineLoc.csv")

        self.imgsDir, self.idxRoom, self.idxEnv, self.coordsX, self.coordsY = \
            testCSV['Img Test'], testCSV['Idx Room'], testCSV['Idx Env'], testCSV['Coord X'], testCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxEnv, idxRoom, coordX, coordY = \
            self.imgsDir[index], self.idxEnv[index], self.idxRoom[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, idxRoom, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TrainGlobalLoc(Dataset):

    def __init__(self, tree, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        trainCSV = pd.read_csv(csvDir + '/Exp2TrainGlobalLoc.csv')

        self.imgsDir, self.idxEnv, self.coordX, self.coordY \
            = trainCSV['Img'], trainCSV["IdxEnv"], trainCSV['Coord X'], trainCSV['Coord Y']

        self.tree = tree
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        rPos, rNeg = PARAMETERS.rPosGlobalLoc, PARAMETERS.rNegGlobalLoc

        idxAnc = index % len(self.imgsDir)
        imgAnc = self.imgsDir[idxAnc]
        coordsImgAnc = np.array([self.coordX[idxAnc], self.coordY[idxAnc]])

        posIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rPos)[0].reshape(1, -1)[0]
        d = 0
        posExamplesEmpty = True
        for idx in posIdxs:
            env = self.idxEnv[idx]
            if env == self.idxEnv[idxAnc] and idx != posIdxs[0]:
                posExamplesEmpty = False
                break

        while len(posIdxs) <= 1 and posExamplesEmpty:
            d += 0.01
            posIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rPos + d)[0].reshape(1, -1)[0]
            for idx in posIdxs:
                env = self.idxEnv[idx]
                if env == self.idxEnv[idxAnc] and idx != posIdxs[0]:
                    posExamplesEmpty = False
                    break

        idxPos = np.random.choice(posIdxs)
        while idxPos == idxAnc:
            idxPos = np.random.choice(posIdxs)
        imgPos = self.imgsDir[idxPos]

        negIdxs = self.tree.query_radius(coordsImgAnc.reshape(1, -1), r=rNeg)[0].reshape(1, -1)[0]

        idxNeg = np.random.choice(range(len(self.imgsDir)))
        while idxNeg in negIdxs:
            idxNeg = np.random.choice(range(len(self.imgsDir)))
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


class VisualModelGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        vmCSV = pd.read_csv(csvDir + '/Exp2VisualModelGlobalLoc.csv')

        self.imgsDir, self.idxEnv, self.coordsX, self.coordsY \
            = vmCSV['Img'], vmCSV['Idx Env'], vmCSV['Coord X'], vmCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxEnv, coordX, coordY = self.imgsDir[index], self.idxEnv[index], self.coordsX[index], self.coordsY[index]

        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class ValidationGlobalLoc(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        valCSV = pd.read_csv(csvDir + '/Exp2ValidationGlobalLoc.csv')

        self.imgsDir, self.idxEnv, self.coordsX, self.coordsY \
            = valCSV['Img'], valCSV['Idx Env'], valCSV['Coord X'], valCSV['Coord Y']

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        img, idxEnv, coordX, coordY = self.imgsDir[index], self.idxEnv[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)


class TestGlobalLoc(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor(), should_invert=False):

        testCSV = pd.read_csv(csvDir + "/Exp2Test" + illumination + "GlobalLoc.csv")

        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

        self.imgsDir, self.idxEnv, self.coordsX, self.coordsY \
            = testCSV['Img'], testCSV['Idx Env'], testCSV['Coord X'], testCSV['Coord Y']

    def __getitem__(self, index):

        img, idxEnv, coordX, coordY = self.imgsDir[index], self.idxEnv[index], self.coordsX[index], self.coordsY[index]
        img = Image.open(img).convert("RGB")

        if self.should_invert:
            img = ImageOps.invert(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, idxEnv, np.array([coordX, coordY])

    def __len__(self):
        return len(self.imgsDir)
