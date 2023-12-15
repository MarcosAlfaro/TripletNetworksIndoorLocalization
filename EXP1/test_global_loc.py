"""
TEST CODE: HIERARCHICAL LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the global method

Test dataset:
Cloudy: seq2cloudy2 (2595 images)
Night: seq2night2 (2707 images)
Sunny: seq2sunny2 (2114 images)

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in one step:
    -each test image is compared with the images of the visual model of the entire map
    -the nearest neighbour indicates the retrieved coordinates
"""


import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import csv
from sklearn.neighbors import KDTree
import torchvision.datasets as dset
from torchvision.models import VGG16_Weights

import create_figures
import create_datasets
from config import PARAMETERS


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
figuresDir = os.path.join(baseDir, "FIGURES", "GlobalLoc")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")

trainingDir = os.path.join(datasetDir, "Entrenamiento")
trainingDataset = dset.ImageFolder(root=trainingDir)
rooms = trainingDataset.classes

kMax = PARAMETERS.kGlobalLoc


def get_loss(red):
    _, lf = red.split("netLG_")
    lf, _ = lf.split("m")
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


visualModelDataset = create_datasets.VisualModelGlobalLoc(imageFolderDataset=datasetDir + '/Entrenamiento/',
                                                          transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                        transforms.ToTensor()
                                                                                        ]),
                                                          should_invert=False)

visualModelDataloader = DataLoader(visualModelDataset, shuffle=False, num_workers=0, batch_size=1)

vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)

with open(csvDir + "/ResultsGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Net", "Ilum", "Geom error", "Min error"]
    for k in range(kMax):
        rowCSV.append("Recallk" + str(k+1))
    writer.writerow(rowCSV)

    netDir = os.path.join(baseDir, "SAVED_MODELS", "GlobalLoc")
    losses = os.listdir(netDir)
    bestError, bestNet = 0, ""
    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        margins = os.listdir(lossDir)
        bestLossError, bestLossNet = 0, ""
        for margin in margins:
            marginDir = os.path.join(lossDir, margin)
            testNets = os.listdir(marginDir)
            bestMarginError, bestMarginNet = 0, ""
            for testNet in testNets:
                testDir = os.path.join(marginDir, testNet)
                net = torch.load(testDir).to(device)
                print(f"TEST NETWORK {testDir}")

                """VISUAL MODEL"""

                descriptorsVM, coordsVM = [], []

                for i, groundTruth_data in enumerate(visualModelDataloader, 0):
                    imgVM, coords = groundTruth_data
                    imgVM = imgVM.to(device)
                    output = net(imgVM)
                    output = output.cpu()
                    descriptorsVM.append(output.detach().numpy()[0])
                    coordsVM.append(coords.detach().numpy()[0])

                treeCoordsVM = KDTree(coordsVM, leaf_size=2)
                treeDesc = KDTree(descriptorsVM, leaf_size=2)

                """
                
                
                
                
                
                """

                condIlum = ['Cloudy', 'Night', 'Sunny']
                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    print(f"Test {ilum}\n")

                    testDataset = create_datasets.TestGlobalLoc(illumination=ilum,
                                                                imageFolderDataset=datasetDir + "/Test" + ilum + "/",
                                                                transform=transforms.Compose([
                                                                          transforms.Resize((128, 512)),
                                                                          transforms.ToTensor()]), should_invert=False)

                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    recallLG = np.zeros((4, kMax))
                    geomError = np.zeros(4)
                    minErrorPossible = np.zeros(4)

                    coordsMapTest = []

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, coordsImgTest = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest)
                        output = output.cpu()
                        output = output.detach().numpy()[0]
                        coordsImgTest = coordsImgTest.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':
                            cosMax = 0
                            for descVM in descriptorsVM:
                                cosSimilarity = np.dot(descVM, output)
                                if cosSimilarity > cosMax:
                                    cosMax = cosSimilarity
                                    idxMinPred = descriptorsVM.index(descVM)
                        else:
                            _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                            idxMinPred = idxDesc[0][0]

                        geomDistances, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=kMax)
                        idxMinReal = idxGeom[0][0]

                        coordsPredictedImg = coordsVM[idxMinPred]
                        coordsClosestImg = coordsVM[idxMinReal]

                        if idxMinPred in idxGeom[0]:
                            label = str(idxGeom[0].tolist().index(idxMinPred)+1)
                            recallLG[idxIlum][idxGeom[0].tolist().index(idxMinPred):] += 1
                        else:
                            label = "F"

                        coordsMapTest.append([coordsPredictedImg[0], coordsPredictedImg[1],
                                              coordsImgTest[0], coordsImgTest[1], label])

                        geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                        minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)

                    recallLG[idxIlum] *= 100 / len(testDataloader)
                    geomError[idxIlum] /= len(testDataloader)
                    minErrorPossible[idxIlum] /= len(testDataloader)

                    create_figures.display_coord_map(figuresDir, 'Global', coordsVM, coordsMapTest, kMax, ilum, loss)

                    print(f"Geometric error: {geomError[idxIlum]} m")
                    print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                    rowCSV = [testNet, ilum, geomError[idxIlum], minErrorPossible[idxIlum]]
                    for k in range(kMax):
                        print(f"Recall@{k+1}: {recallLG[idxIlum][k]}%")
                        rowCSV.append(recallLG[idxIlum][k])
                    print(f"\n")
                    writer.writerow(rowCSV)

                geomError[3] = (geomError[0] + geomError[1] + geomError[2]) / 3
                recallLG[3] = (recallLG[0] + recallLG[1] + recallLG[2]) / 3
                minErrorPossible[3] = (minErrorPossible[0] + minErrorPossible[1] + minErrorPossible[2]) / 3

                avgGeomError = geomError[3]
                avgRecallLG = recallLG[3]
                avgMinError = minErrorPossible[3]
                rowCSV = [testNet, "Average", avgGeomError, avgMinError]
                for k in range(kMax):
                    rowCSV.append(avgRecallLG[k])
                writer.writerow(rowCSV)

                if avgGeomError > bestMarginError:
                    bestMarginNet = testNet
                    bestMarginError = avgGeomError
                if avgGeomError > bestLossError:
                    bestLossNet = testNet
                    bestLossError = avgGeomError
                if avgGeomError > bestError:
                    bestNet = testNet
                    bestError = avgGeomError

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Geometric Error: {bestMarginError} m")
        if bestLossNet != "":
            print(f"Best net loss {loss}: {bestLossNet}, Geometric Error: {bestLossError} m")
    if bestNet != "":
        print(f"Best net: {bestNet}, Geometric error: {bestError} m")
