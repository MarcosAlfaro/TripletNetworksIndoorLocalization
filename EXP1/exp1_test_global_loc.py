"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to test triplet networks to perform the global localization

Test dataset:
Cloudy: seq2cloudy2 (2595 images), Night: seq2night2 (2707 images), Sunny: seq2sunny2 (2114 images)

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in one step:
    -each test image is compared with the images of the visual model of the entire map
    -the nearest neighbor indicates the retrieved coordinates
    -metric: geometric error (m)

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: figuresDir*, datasetDir*, csvDir*, modelsDir*
*keep the same for all the scripts
"""


import os
import csv
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.neighbors import KDTree
import create_figures
import exp1_create_datasets
from functions import create_path
from config import PARAMS


device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = os.path.join(PARAMS.csvDir, "RESULTS")
figuresDir = create_path(os.path.join(PARAMS.figuresDir, "EXP1", "GlobalLoc"))

condIlum = ['Cloudy', 'Night', 'Sunny']

vmDataset = exp1_create_datasets.VisualModel()
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)


with open(csvDir + "/Exp1GlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Loss", "Margin", "Net"]
    for ilum in condIlum:
        rowCSV.append(ilum + " Error")
    rowCSV.append("Average Error")
    writer.writerow(rowCSV)

    netDir = os.path.join(PARAMS.modelsDir, "EXP1", "GlobalLoc")
    losses = os.listdir(netDir)
    bestError, bestLoss = 100, ""
    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        margins = os.listdir(lossDir)
        bestLossError, bestMargin = 100, ""
        for margin in margins:
            marginDir = os.path.join(lossDir, margin)
            testNets = os.listdir(marginDir)
            bestMarginError, bestNet = 100, ""
            print(f"Test {loss}, m={margin}\n")
            for testNet in testNets:
                testDir = os.path.join(marginDir, testNet)
                net = torch.load(testDir).to(device)
                net.eval()

                with torch.no_grad():

                    print(f"Test {testNet}\n")

                    """VISUAL MODEL"""

                    descVM, coordsVM, roomsVM = [], [], []
                    for i, vmData in enumerate(vmDataloader, 0):
                        imgVM, idxRoom, coords = vmData
                        imgVM = imgVM.to(device)
                        output = net(imgVM).cpu().detach().numpy()[0]
                        roomsVM.append(idxRoom.detach().numpy()[0])
                        descVM.append(output)
                        coordsVM.append(coords.detach().numpy()[0])
                    treeCoordsVM = KDTree(coordsVM, leaf_size=2)
                    treeDesc = KDTree(descVM, leaf_size=2)


                    """TEST"""

                    geomError, minErrorPossible = np.zeros(len(condIlum)), np.zeros(len(condIlum))
                    recall = np.zeros((len(condIlum), PARAMS.kMax))

                    for ilum in condIlum:
                        idxIlum = condIlum.index(ilum)

                        print(f"Test {ilum}")

                        testDataset = exp1_create_datasets.Test(illumination=ilum)
                        testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                        coordsMapTest = []

                        for i, data in enumerate(testDataloader, 0):
                            imgTest, _, coordsImgTest = data
                            imgTest = imgTest.to(device)

                            output = net(imgTest).cpu().detach().numpy()[0]
                            coordsImgTest = coordsImgTest.detach().numpy()[0]

                            if loss in ['CL', 'AL']:
                                cosSimilarities = np.dot(descVM, output)
                                idxMinPred = np.argmax(cosSimilarities)
                            else:
                                _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                                idxMinPred = idxDesc[0][0]

                            geomDistances, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=PARAMS.kMax)
                            idxMinReal = idxGeom[0][0]

                            coordsPredictedImg, coordsClosestImg = coordsVM[idxMinPred], coordsVM[idxMinReal]

                            if idxMinPred in idxGeom[0]:
                                label = str(idxGeom[0].tolist().index(idxMinPred) + 1)
                                recall[idxIlum][idxGeom[0].tolist().index(idxMinPred):] += 1
                            else:
                                label = "F"

                            coordsMapTest.append([coordsPredictedImg[0], coordsPredictedImg[1],
                                                  coordsImgTest[0], coordsImgTest[1], label])

                            geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                            minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)

                        recall[idxIlum] *= 100 / len(testDataloader)
                        geomError[idxIlum] /= len(testDataloader)
                        minErrorPossible[idxIlum] /= len(testDataloader)

                        create_figures.display_coord_map(figuresDir, "Global", coordsVM, coordsMapTest,
                                                         PARAMS.kMax, ilum, loss)

                        print(f"Geometric error: {geomError[idxIlum]} m, "
                              f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                    avgGeomError, avgMinError = np.average(geomError), np.average(minErrorPossible)
                    avgRecall = np.average(recall, axis=0)

                    writer.writerow([loss, margin, testNet, geomError[0], geomError[1], geomError[2],
                                     recall[idxIlum][0], recall[idxIlum][round(len(vmDataloader) / 100)]])

                    if avgGeomError < bestMarginError:
                        bestNet, bestMarginError = testNet, avgGeomError
                        if avgGeomError < bestLossError:
                            bestMargin, bestLossError = margin, avgGeomError
                            if avgGeomError < bestError:
                                bestLoss, bestError = loss, avgGeomError

            if bestNet != "":
                print(f"Best Net {loss}, margin {margin}: {bestNet}, Geometric error: {bestMarginError} m")
        if bestMargin != "":
            print(f"Best Margin {loss}: {margin}, Geometric error: {bestLossError} m")
    if bestLoss != "":
        print(f"Best Loss: {bestLoss}, Geometric error: {bestError} m")
