"""
TEST CODE: GLOBAL LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the global method

In this experiment, three different environments are considered: Friburgo, Saarbrücken A & Saarbrücken B

Test dataset:
- Cloudy:
    - Friburgo: seq2cloudy2 Sampled (867 images)
    - Saarbrücken A: seq2cloudy2 Sampled (758 images)
    - Saarbrücken B: seq4cloudy2 Sampled (281 images)
- Night
    - Friburgo: seq2night2 Sampled (905 images)
    - Saarbrücken A: seq2night1 Sampled (759 images)
    - Saarbrücken B: seq4night2 Sampled (292 images)
- Sunny
    - Friburgo: seq2cloudy2 Sampled (707 images)
    - Saarbrücken A: X
    - Saarbrücken B: seq4cloudy2 Sampled (281 images)

Visual model dataset: the training set is employed as visual model

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
import create_figures2
import create_datasets2
from config import PARAMETERS


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
figuresDir = os.path.join(baseDir, "FIGURES", "EXP2", "GlobalLoc")
datasetDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS")

condIlum = ['Cloudy', 'Night', 'Sunny']
kMax = PARAMETERS.kGlobalLoc


vmDataset = create_datasets2.VisualModelGlobalLoc(imageFolderDataset=datasetDir + '/Train/')
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)


with open(csvDir + "/Exp2ResultsGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Net", "Ilum", "Geom error", "Min error"]
    for k in range(kMax):
        rowCSV.append("Recallk" + str(k+1))
    writer.writerow(rowCSV)

    netDir = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "GlobalLoc")
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

                descriptorsVM, coordsVM, idxsEnv = [], [], []
                coordsVMenv = [[], [], []]
                for i, vmData in enumerate(vmDataloader, 0):
                    imgVM, idxEnvVM, coordsImgEnv = vmData

                    imgVM = imgVM.to(device)
                    output = net(imgVM).cpu().detach().numpy()[0]
                    descriptorsVM.append(output)

                    idxEnvVM = idxEnvVM.detach().numpy()[0]
                    idxsEnv.append(idxEnvVM)

                    coordsImgEnv = coordsImgEnv.detach().numpy()[0]
                    coordsVM.append(coordsImgEnv)
                    coordsVMenv[idxEnvVM].append(coordsImgEnv)

                treeDesc = KDTree(descriptorsVM, leaf_size=2)
                treeCoordsVMenv = []
                for env in range(3):
                    treeCoordsVMenv.append(KDTree(coordsVMenv[env], leaf_size=2))

                """
                
                
                
                
                
                """

                recallLG = np.zeros((len(condIlum), kMax))
                geomError = np.zeros(len(condIlum))
                minErrorPossible = np.zeros(len(condIlum))
                accuracyEnv = np.zeros(len(condIlum))

                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)
                    print(f"Test {ilum}\n")

                    testDataset = create_datasets2.TestGlobalLoc(
                        illumination=ilum, imageFolderDataset=datasetDir + "/Test" + ilum + "/")
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, actualEnv, coordsImgTest = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]
                        coordsImgTest = coordsImgTest.detach().numpy()[0]
                        actualEnv = actualEnv.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':
                            cosSimilarities = np.dot(descriptorsVM, output)
                            idxMinPred = np.argmax(cosSimilarities)
                        else:
                            _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                            idxMinPred = idxDesc[0][0]

                        geomDistances, idxGeom = treeCoordsVMenv[actualEnv].query(coordsImgTest.reshape(1, -1), k=kMax)
                        idxMinReal = idxGeom[0][0]

                        coordsPredictedImg = coordsVM[idxMinPred]
                        coordsClosestImg = coordsVMenv[actualEnv][idxMinReal]

                        if idxMinPred in idxGeom[0]:
                            recallLG[idxIlum][idxGeom[0].tolist().index(idxMinPred):] += 1

                        predictedEnv = idxsEnv[idxMinPred]
                        if predictedEnv == actualEnv:
                            accuracyEnv[idxIlum] += 1
                            geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                        minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)

                    recallLG[idxIlum] *= 100 / len(testDataloader)
                    geomError[idxIlum] /= accuracyEnv[idxIlum]
                    minErrorPossible[idxIlum] /= len(testDataloader)

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
