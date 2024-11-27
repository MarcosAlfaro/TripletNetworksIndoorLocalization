"""
EXPERIMENT 3:
-evaluation of triplet networks in different environments simultaneously

This script is used to test triplet networks to perform the coarse localization (first stage of hierarchical loc.)

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


-each test image is compared with the representative image of every room
-the closest representative descriptor indicates the retrieved room
- the environment accuracy is as well analyzed

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: figuresDir*, datasetDir*, csvDir*, modelsDir*
*keep the same for all the scripts
"""

import os
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree
import exp3_create_datasets
from functions import get_env
from config import PARAMS


device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

csvDir = os.path.join(PARAMS.csvDir, "RESULTS")
datasetDir = os.path.join(PARAMS.datasetDir, "3ENVIRONMENTS")

condIlum = ['Cloudy', 'Night', 'Sunny']


imgRepDataset = exp3_create_datasets.RepImages()
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)


with open(csvDir + "/Exp3CoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Cloudy Accuracy", "Night Accuracy", "Sunny Accuracy", "Average Accuracy"])

    netDir = os.path.join(PARAMS.modelsDir, "EXP3", "HierarchicalLoc", "CoarseLoc")
    losses = os.listdir(netDir)
    bestAccuracy, bestNet = 0, ""
    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        margins = os.listdir(lossDir)
        bestLossAccuracy, bestLossNet = 0, ""
        for margin in margins:
            marginDir = os.path.join(lossDir, margin)
            testNets = os.listdir(marginDir)
            bestMarginAccuracy, bestMarginNet = 0, ""
            for testNet in testNets:
                testDir = os.path.join(marginDir, testNet)
                net = torch.load(testDir).to(device)
                print(f"Test {testNet}\n")

                """REPRESENTATIVE IMAGES"""

                descImgRep, descImgRepSunny = [], []
                for i, imgRepData in enumerate(imgRepDataloader, 0):
                    imgRep, room = imgRepData[0].to(device), imgRepData[1]
                    output = net(imgRep).cpu().detach().numpy()[0]
                    descImgRep.append(output)
                    if not 'SA-A' in room[0]:
                        descImgRepSunny.append(output)
                treeImgRep = KDTree(descImgRep, leaf_size=2)
                treeImgRepSunny = KDTree(descImgRepSunny, leaf_size=2)

                roomAccuracy, envAccuracy = np.zeros(len(condIlum)), np.zeros(len(condIlum))

                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    print(f"Test {ilum}")
                    testDataset = exp3_create_datasets.Test(illumination=ilum)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    ilumDataset = dset.ImageFolder(root=datasetDir + "/Test" + ilum + "/")
                    rooms = ilumDataset.classes

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, actualRoom, _ = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]

                        actualRoom = actualRoom[0]
                        actualEnv = get_env(actualRoom)

                        if ilum == "Sunny":
                            repDescriptors = descImgRepSunny[:]
                            tree = treeImgRepSunny
                        else:
                            repDescriptors = descImgRep[:]
                            tree = treeImgRep

                        if loss in ['CL', 'AL']:
                            cosSimilarities = np.dot(repDescriptors, output)
                            idxRoom = np.argmax(cosSimilarities)
                        else:
                            _, idxRoom = tree.query(output.reshape(1, -1), k=1)
                            idxRoom = idxRoom[0][0]
                        predictedRoom = rooms[idxRoom]
                        predictedEnv = get_env(predictedRoom)

                        if predictedEnv == actualEnv:
                            envAccuracy[idxIlum] += 1
                            if predictedRoom == actualRoom:
                                roomAccuracy[idxIlum] += 1

                    roomAccuracy[idxIlum] *= 100 / len(testDataloader)
                    envAccuracy[idxIlum] *= 100 / len(testDataloader)

                    print(f'Environment accuracy: {envAccuracy[idxIlum]} %')
                    print(f'Room accuracy: {roomAccuracy[idxIlum]} %\n')

                avgEnvAccuracy, avgRoomAccuracy = np.average(envAccuracy), np.average(roomAccuracy)

                if avgRoomAccuracy > bestMarginAccuracy:
                    bestMarginNet = testNet
                    bestMarginAccuracy = avgRoomAccuracy
                    if avgRoomAccuracy > bestLossAccuracy:
                        bestMargin = testNet
                        bestLossAccuracy = avgRoomAccuracy
                        if avgRoomAccuracy > bestAccuracy:
                            bestLoss = loss
                            bestAccuracy = avgRoomAccuracy

                print(f'Average:')
                print(f'Environment accuracy: {avgEnvAccuracy} %')
                print(f'Room accuracy: {avgRoomAccuracy} %\n')

                writer.writerow([loss, margin, roomAccuracy[0], roomAccuracy[1], roomAccuracy[2], avgRoomAccuracy])

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Accuracy: {bestMarginAccuracy} %")
        if bestMargin != "":
            print(f"Best net loss {loss}: {bestLossNet}, Accuracy: {bestLossAccuracy} %")
    if bestLoss != "":
        print(f"Best loss: {bestLoss}, Accuracy: {bestAccuracy} %")
