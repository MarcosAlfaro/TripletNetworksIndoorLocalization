"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to test triplet networks to perform the coarse localization (first stage of hierarchical loc.)

Test dataset:
Cloudy: seq2cloudy2 (2595 images), Night: seq2night2 (2707 images), Sunny: seq2sunny2 (2114 images)

-each test image is compared with the representative image of every room
-the closest representative descriptor indicates the retrieved room
- the metric used is the room retrieval accuracy (%)

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: figuresDir*, datasetDir*, csvDir*, modelsDir*
*keep the same for all the scripts
"""

import os
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree
import exp1_create_datasets
from config import PARAMS

device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = os.path.join(PARAMS.csvDir, "RESULTS")
condIlum = ['Cloudy', 'Night', 'Sunny']


imgRepDataset = exp1_create_datasets.RepImages()
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

with open(csvDir + "/Exp1CoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Loss", "Margin", "Net"]
    for ilum in condIlum:
        rowCSV.append(ilum)
    rowCSV.append("Average")
    writer.writerow(rowCSV)

    netDir = os.path.join(PARAMS.modelsDir, "EXP1", "HierarchicalLoc", "CoarseLoc")
    losses = os.listdir(netDir)
    bestAccuracy, bestLoss = 0, ""
    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        margins = os.listdir(lossDir)
        bestLossAccuracy, bestMargin = 0, ""
        for margin in margins:
            print(f"\n\nTEST LOSS {loss}, m={margin}\n")
            marginDir = os.path.join(lossDir, margin)
            testNets = os.listdir(marginDir)
            bestMarginAccuracy, bestNet = 0, ""
            for testNet in testNets:
                testDir = os.path.join(marginDir, testNet)
                net = torch.load(testDir).to(device)

                net.eval()
                print(f"Test {testNet}")
                rowCSV = [loss, margin, testNet]

                with torch.no_grad():
                    """REPRESENTATIVE IMAGES"""

                    descImgRep = []
                    for i, imgRepData in enumerate(imgRepDataloader, 0):
                        imgRep = imgRepData[0].to(device)
                        output = net(imgRep).cpu().detach().numpy()[0]
                        descImgRep.append(output)
                    treeImgRep = KDTree(descImgRep, leaf_size=2)

                    accuracy = np.zeros(len(condIlum))
                    for ilum in condIlum:
                        print(f"Test {ilum}\n")
                        idxIlum = condIlum.index(ilum)

                        testDataset = exp1_create_datasets.Test(illumination=ilum)
                        testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                        for i, data in enumerate(testDataloader, 0):

                            imgTest, actualRoom, _ = data
                            imgTest = imgTest.to(device)

                            output = net(imgTest).cpu().detach().numpy()[0]
                            actualRoom = actualRoom.detach().numpy()[0]

                            if loss in ['CL', 'AL']:
                                cosSimilarities = np.dot(descImgRep, output)
                                predictedRoom = np.argmax(cosSimilarities)
                            else:
                                _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                                predictedRoom = predictedRoom[0][0]

                            if predictedRoom == actualRoom:
                                accuracy[idxIlum] += 1

                        accuracy[idxIlum] *= 100 / len(testDataloader)
                        print(f'{ilum} accuracy: {accuracy[idxIlum]} %')
                        rowCSV.append(accuracy[idxIlum])

                    avgAccuracy = np.average(accuracy)

                    if avgAccuracy > bestMarginAccuracy:
                        bestNet = testNet
                        bestMarginAccuracy = avgAccuracy
                        if avgAccuracy > bestLossAccuracy:
                            bestMargin = margin
                            bestLossAccuracy = avgAccuracy
                            if avgAccuracy > bestAccuracy:
                                bestLoss = loss
                                bestAccuracy = avgAccuracy

                    print(f'Average accuracy: {avgAccuracy}%\n')
                    rowCSV.append(avgAccuracy)
                    writer.writerow(rowCSV)

            if bestNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestNet}, Accuracy: {bestMarginAccuracy} %")
        if bestMargin != "":
            print(f"Best margin loss {loss}: {bestMargin}, Accuracy: {bestLossAccuracy} %")
    if bestLoss != "":
        print(f"Best loss: {bestLoss}, Accuracy: {bestAccuracy} %")
