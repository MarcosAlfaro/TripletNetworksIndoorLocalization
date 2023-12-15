"""
TEST CODE: HIERARCHICAL LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the hierarchical method

Test dataset:
Cloudy: seq2cloudy2 (2595 images)
Night: seq2night2 (2707 images)
Sunny: seq2sunny2 (2114 images)

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in two steps:
-Coarse step: room retrieval task
    -each test image is compared with the representative image of every room
    -the closest representative descriptor indicates the retrieved room
-Fine step: obtain the coordinates of the robot inside the retrieved room:
    -each test image is compared with the images of the visual model of the retrieved room
    -the nearest neighbour indicates the retrieved coordinates
"""


import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import csv
from torchvision.models import VGG16_Weights
from sklearn.neighbors import KDTree

import create_datasets
import create_figures
from config import PARAMETERS

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
figuresDir = os.path.join(baseDir, "FIGURES", "FineLoc")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")

trainingDir = os.path.join(datasetDir, "Entrenamiento")
trainingDataset = dset.ImageFolder(root=trainingDir)
rooms = trainingDataset.classes

kMax = PARAMETERS.kFineLoc


def get_loss(red):
    _, lf = red.split("netLf_")
    lf, _ = lf.split("_ep")
    lf, _ = lf.split("_")
    return lf


class TripletNetwork(nn.Module):

    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn1 = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * 512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def forward_once(self, x):
        verbose = False

        if verbose:
            print("Input: ", x.size())

        out = self.cnn1(x)

        if verbose:
            print("Output matricial: ", out.size())

        out = self.avgpool(out)
        if verbose:
            print("Output avgpool: ", out.size())
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        norm = True
        if norm:
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out

    def forward(self, input1):
        out1 = self.forward_once(input1)
        return out1


vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)

imgRepDataset = create_datasets.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/",
                                                     transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                   transforms.ToTensor()
                                                                                   ]),
                                                     should_invert=False)

imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

visualModelDataset = create_datasets.VisualModelTestFineLoc(imageFolderDataset=datasetDir + "/Entrenamiento/",
                                                            transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                          transforms.ToTensor()
                                                                                          ]),
                                                            should_invert=False)

visualModelDataloader = DataLoader(visualModelDataset, shuffle=False, num_workers=0, batch_size=1)


with open(csvDir + '/ResultsHierarchicalLoc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    columnsCSV = ["Loss", "Ilum", "CoarseLoc Accuracy"]
    for room in rooms:
        columnsCSV.append('Geom Error ' + room)
    columnsCSV.append('Geom Error')
    for k in range(kMax):
        columnsCSV.append('Recallk' + str(k+1))

    writer.writerow(columnsCSV)
    firstRow = [True, True, True]

    netDir = os.path.join(baseDir, "SAVED_MODELS", "FineLoc")
    losses = PARAMETERS.lossesFineLocTest

    for loss in losses:
        # these networks must be renamed and copied in directory "baseDir/SAVED_MODELS/FineLoc/loss/" manually
        idxLoss = PARAMETERS.losses.index(loss)
        sl = PARAMETERS.lossAbreviations[idxLoss]
        testNetLg = os.path.join(baseDir, "SAVED_MODELS", "CoarseLoc", sl, "netLg_" + sl)
        lossDir = os.path.join(netDir, sl)
        allNetsLf = []
        numMaxNets = 0
        bestErrorRooms = 100 * np.ones(len(rooms))
        bestNets = []
        for room in rooms:
            allNetsLf.append([])
            roomDir = os.path.join(lossDir, room)
            for roomNet in os.listdir(roomDir):
                allNetsLf[rooms.index(room)].append(roomNet)
            bestNets.append("")
            if len(os.listdir(roomDir)) > numMaxNets:
                numMaxNets = len(os.listdir(roomDir))
        for n in range(numMaxNets):
            testNetsLf = []
            for room in rooms:
                if n < len(allNetsLf[rooms.index(room)]):
                    testNetsLf.append(allNetsLf[rooms.index(room)][n])
                else:
                    testNetsLf.append(allNetsLf[rooms.index(room)][-1])

            netLg = torch.load(testNetLg).to(device)

            print(f"TEST HIERARCHICAL LOCALIZATION\n Loss: {loss}\n\n")

            """REPRESENTATIVE IMAGES"""

            descImgRep = []
            for i, imgRepData in enumerate(imgRepDataloader, 0):
                imgRep, _ = imgRepData
                imgRep = imgRep.to(device)

                output = netLg(imgRep)
                output = output.cpu()

                descImgRep.append(output.detach().numpy()[0])

            treeImgRep = KDTree(descImgRep, leaf_size=2)

            """VISUAL MODEL"""

            coordsVM, coordsVMrooms = [], []
            descriptorsVM = []
            treeDescVMrooms = []

            for room in rooms:
                idxRoom = rooms.index(room)
                testNetDir = os.path.join(baseDir, "SAVED_MODELS", "FineLoc", sl, room, testNetsLf[idxRoom])
                netLf = torch.load(testNetDir)
                descVMroom, coordsVMroom = [], []
                for i, VMdata in enumerate(visualModelDataloader, 0):

                    imgVM, ind_gt, coords = VMdata
                    imgVM = imgVM.to(device)

                    if ind_gt.detach().numpy()[0] == idxRoom:
                        output = netLf(imgVM)
                        output = output.cpu()
                        descVMroom.append(output.detach().numpy()[0])
                        coordsVMroom.append(coords.detach().numpy()[0])
                        coordsVM.append(coords.detach().numpy()[0])

                coordsVMrooms.append(coordsVMroom)
                descriptorsVM.append(descVMroom)
                treeDescVMrooms.append(KDTree(descVMroom, leaf_size=2))

            treeCoordsVM = KDTree(coordsVM, leaf_size=2)
            """
        
        
        
        
        
        
            """

            condIlum = ['Cloudy', 'Night', 'Sunny']

            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)

                print(f"Test {ilum}\n")

                testDataset = create_datasets.TestFineLoc(illumination=ilum,
                                                          imageFolderDataset=datasetDir + "/Test" + ilum + "/",
                                                          transform=transforms.Compose([
                                                                    transforms.Resize((128, 512)),
                                                                    transforms.ToTensor()]), should_invert=False)

                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                accuracyCoarseLoc = np.zeros(4)
                recallLF = np.zeros((4, kMax))
                geomError, minErrorPossible = np.zeros(4), np.zeros(4)
                geomErrorRooms, minErrorRooms = np.zeros((4, len(rooms))), np.zeros((4, len(rooms)))

                coordsMapTest = []

                actualRooms, predRooms = [], []

                for i, data in enumerate(testDataloader, 0):
                    imgTest, actualRoom, coordsImgTest = data
                    imgTest = imgTest.to(device)

                    """COARSE LOCALIZATION"""

                    actualRoom = actualRoom.detach().numpy()[0]

                    output = netLg(imgTest)
                    output = output.cpu()
                    output = output.detach().numpy()[0]

                    if sl == 'CL' or sl == 'AL':

                        cosMax = 0
                        for desc in descImgRep:
                            cosSimilarity = np.dot(desc, output)
                            if cosSimilarity > cosMax:
                                cosMax = cosSimilarity
                                predictedRoom = descImgRep.index(desc)
                    else:
                        _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                        predictedRoom = predictedRoom[0][0]

                    if predictedRoom == actualRoom:
                        accuracyCoarseLoc[idxIlum] += 1

                    actualRooms.append(actualRoom)
                    predRooms.append(predictedRoom)

                    """FINE LOCALIZATION"""

                    testNetDir = os.path.join(baseDir, "SAVED_MODELS", "FineLoc", sl, rooms[predictedRoom],
                                              testNetsLf[predictedRoom])
                    netLf = torch.load(testNetDir)

                    treeDescVM = treeDescVMrooms[predictedRoom]

                    output = netLf(imgTest)
                    output = output.cpu()
                    output = output.detach().numpy()[0]
                    coordsImgTest = coordsImgTest.detach().numpy()[0]

                    if sl == 'CL' or sl == 'AL':
                        cosMax = 0
                        for desc in descriptorsVM[predictedRoom]:
                            cosSimilarity = np.dot(desc, output)
                            if cosSimilarity > cosMax:
                                cosMax = cosSimilarity
                                idxMinPred = descriptorsVM[predictedRoom].index(desc)
                    else:
                        _, idxDesc = treeDescVM.query(output.reshape(1, -1), k=1)
                        idxMinPred = idxDesc[0][0]

                    geomDistances, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=kMax)
                    idxMinReal = idxGeom[0][0]

                    coordsPredictedImg = coordsVMrooms[predictedRoom][idxMinPred]
                    # coordsClosestImg = coordsVMrooms[predictedRoom][idxMinReal]
                    coordsClosestImg = coordsVM[idxMinReal]

                    if actualRoom != predictedRoom:
                        label = "R"
                        if idxMinPred in idxGeom[0]:
                            recallLF[idxIlum][idxGeom[0].tolist().index(idxMinPred):] += 1
                    elif idxMinPred in idxGeom[0]:
                        label = str(idxGeom[0].tolist().index(idxMinPred)+1)
                        recallLF[idxIlum][idxGeom[0].tolist().index(idxMinPred):] += 1
                    else:
                        label = "F"
                    coordsMapTest.append([coordsPredictedImg[0], coordsPredictedImg[1],
                                          coordsImgTest[0], coordsImgTest[1], label])

                    geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                    geomErrorRooms[idxIlum][actualRoom] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                    minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)
                    minErrorRooms[idxIlum][actualRoom] += np.linalg.norm(coordsImgTest - coordsClosestImg)

                for room in range(len(rooms)):
                    auxDir = os.path.join(datasetDir, "Test" + ilum, rooms[room])
                    geomErrorRooms[idxIlum][room] /= len(os.listdir(auxDir))
                    minErrorRooms[idxIlum][room] /= len(os.listdir(auxDir))

                accuracyCoarseLoc[idxIlum] *= 100 / len(testDataloader)
                recallLF[idxIlum] *= 100 / len(testDataloader)
                geomError[idxIlum] /= len(testDataloader)
                minErrorPossible[idxIlum] /= len(testDataloader)
                if firstRow[idxIlum]:
                    row = ["Loss", "Average", 100]
                    for room in range(len(rooms)):
                        row.append(minErrorRooms[idxIlum][room])
                    row.append(minErrorPossible[idxIlum])
                    for k in range(kMax):
                        row.append(100)
                    writer.writerow([row])
                    firstRow[idxIlum] = False

                create_figures.display_coord_map(figuresDir, 'Hierarchical',
                                                 coordsVM, coordsMapTest, kMax, ilum, loss)

                create_figures.display_confusion_matrix(actual=actualRooms, predicted=predRooms,
                                                        rooms=rooms, plt_name=os.path.join(figuresDir, 'cm' + sl +
                                                                                           '_' + ilum + '.png'),
                                                        loss=loss, ilum=ilum)

                print(f"COARSE LOC\nAccuracy: {accuracyCoarseLoc[idxIlum]}%\n")
                print(f"FINE LOC")

                print(f"Geometric error: {geomError[idxIlum]} m")
                print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                row = [sl, ilum, accuracyCoarseLoc[idxIlum]]
                for room in range(len(rooms)):
                    row.append(geomErrorRooms[idxIlum][room])
                row.append(geomError[idxIlum])
                for k in range(kMax):
                    print(f"Recall@{k+1}: {recallLF[idxIlum][k]}%")
                    row.append(recallLF[idxIlum][k])
                print("\n")
                writer.writerow([row])

            accuracyCoarseLoc[3] = (accuracyCoarseLoc[0] + accuracyCoarseLoc[1] + accuracyCoarseLoc[2]) / 3
            geomError[3] = (geomError[0] + geomError[1] + geomError[2]) / 3
            minErrorPossible[3] = (minErrorPossible[0] + minErrorPossible[1] + minErrorPossible[2]) / 3
            geomErrorRooms[3] = (geomErrorRooms[0] + geomErrorRooms[1] + geomErrorRooms[2]) / 3
            minErrorRooms[3] = (minErrorRooms[0] + minErrorRooms[1] + minErrorRooms[2]) / 3
            recallLF[3] = (recallLF[0] + recallLF[1] + recallLF[2]) / 3

            create_figures.error_rooms(figuresDir, geomError, minErrorPossible,
                                       geomErrorRooms, minErrorRooms, rooms, loss)

            row = [sl, ilum, accuracyCoarseLoc[3]]
            for room in range(len(rooms)):
                row.append(geomErrorRooms[3][room])
            row.append(geomError[3])
            for k in range(kMax):
                row.append(recallLF[3][k])
            writer.writerow([row])

            if geomErrorRooms[3][room] < bestErrorRooms[room]:
                bestErrorRooms[room] = geomErrorRooms[3][room]
                bestNets[room] = testNetsLf[room]

        for room in rooms:
            print(f"Best net loss {loss}, room {room}: {bestNets[rooms.index[room]]},"
                  f" Geometric Error: {bestErrorRooms[rooms.index[room]]} m")
