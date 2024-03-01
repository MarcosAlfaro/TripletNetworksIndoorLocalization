"""
TEST CODE: GLOBAL LOC

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
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree
import create_figures
import create_datasets
from config import PARAMETERS


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
figuresDir = os.path.join(baseDir, "FIGURES", "EXP1", "GlobalLoc")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")

kMax = PARAMETERS.kGlobalLoc

condIlum = ['Cloudy', 'Night', 'Sunny']

vmDataset = create_datasets.VisualModelGlobalLoc(imageFolderDataset=datasetDir + '/Train/')
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)


with open(csvDir + "/Exp1ResultsGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Net", "Ilum", "Geom error", "Min error"]
    for k in range(kMax):
        rowCSV.append("Recall@k" + str(k+1))
    writer.writerow(rowCSV)

    netDir = os.path.join(baseDir, "SAVED_MODELS", "EXP1", "GlobalLoc")
    losses = os.listdir(netDir)
    bestError, bestNet = 100, ""
    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        margins = os.listdir(lossDir)
        bestLossError, bestLossNet = 100, ""
        for margin in margins:
            marginDir = os.path.join(lossDir, margin)
            testNets = os.listdir(marginDir)
            bestMarginError, bestMarginNet = 100, ""
            for testNet in testNets:
                testDir = os.path.join(marginDir, testNet)
                net = torch.load(testDir).to(device)
                print(f"TEST NETWORK {testDir}")

                """VISUAL MODEL"""

                descriptorsVM, coordsVM = [], []

                for i, vmData in enumerate(vmDataloader, 0):
                    imgVM, coords = vmData
                    imgVM = imgVM.to(device)

                    output = net(imgVM).cpu().detach().numpy()[0]
                    descriptorsVM.append(output)
                    coordsVM.append(coords.detach().numpy()[0])
                treeCoordsVM = KDTree(coordsVM, leaf_size=2)
                treeDesc = KDTree(descriptorsVM, leaf_size=2)

                """
                
                
                
                
                
                """

                recallLG = np.zeros((len(condIlum), kMax))
                geomError = np.zeros(len(condIlum))
                minErrorPossible = np.zeros(len(condIlum))

                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    print(f"Test {ilum}\n")

                    testDataset = create_datasets.TestGlobalLoc(
                        illumination=ilum, imageFolderDataset=datasetDir + "/Test" + ilum + "/")
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    coordsMapTest = []

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, coordsImgTest = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]
                        coordsImgTest = coordsImgTest.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':
                            cosSimilarities = np.dot(descriptorsVM, output)
                            predictedRoom = np.argmax(cosSimilarities)
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

                    create_figures.display_coord_map(figuresDir, 'Global', coordsVM,
                                                     coordsMapTest, kMax, ilum, loss)

                    print(f"Geometric error: {geomError[idxIlum]} m")
                    print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                    rowCSV = [testNet, ilum, geomError[idxIlum], minErrorPossible[idxIlum]]
                    for k in range(kMax):
                        print(f"Recall@{k+1}: {recallLG[idxIlum][k]} %")
                        rowCSV.append(recallLG[idxIlum][k])
                    print(f"\n")
                    writer.writerow(rowCSV)

                avgGeomError = np.average(geomError)
                avgRecallLG = np.average(recallLG, axis=0)
                avgMinError = np.average(minErrorPossible)

                rowCSV = [testNet, "Average", avgGeomError, avgMinError]
                for k in range(kMax):
                    rowCSV.append(avgRecallLG[k])
                writer.writerow(rowCSV)

                if avgGeomError < bestMarginError:
                    bestMarginNet = testNet
                    bestMarginError = avgGeomError
                    if avgGeomError < bestLossError:
                        bestLossNet = testNet
                        bestLossError = avgGeomError
                        if avgGeomError < bestError:
                            bestNet = testNet
                            bestError = avgGeomError

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Geometric Error: {bestMarginError} m")
        if bestLossNet != "":
            print(f"Best net loss {loss}: {bestLossNet}, Geometric Error: {bestLossError} m")
    if bestNet != "":
        print(f"Best net: {bestNet}, Geometric error: {bestError} m")
