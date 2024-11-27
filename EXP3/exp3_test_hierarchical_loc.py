"""
EXPERIMENT 3:
-evaluation of triplet networks in different environments simultaneously

This script is used to test triplet networks to perform the coarse localization (second stage of hierarchical loc.)

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

The test is performed in two steps:
-Coarse step: room retrieval task
    -each test image is compared with the representative image of every room
    -the closest representative descriptor indicates the retrieved room
    - the environment accuracy is as well analyzed
-Fine step: obtain the coordinates of the robot inside the retrieved room:
    -each test image is compared with the images of the visual model of the retrieved room
    -the nearest neighbor indicates the retrieved coordinates

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: figuresDir*, datasetDir*, csvDir*, modelsDir*
* keep the same for all the scripts
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
print(f'Using device: {device}')


csvDir = os.path.join(PARAMS.csvDir, "RESULTS")
datasetDir = os.path.join(PARAMS.datasetDir, "3ENVIRONMENTS")


trainDir = os.path.join(datasetDir, "Train")
trainDataset = dset.ImageFolder(root=trainDir)
rooms = trainDataset.classes

condIlum = ['Cloudy', 'Night', 'Sunny']


imgRepDataset = exp3_create_datasets.RepImages()
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

vmDataset = exp3_create_datasets.VisualModel()
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

with open(csvDir + '/Exp3HierarchicalLoc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Ilum", "CoarseLoc Accuracy", "Geom Error", "Recall@1", "Recall@1%"])

    savedModelsDir = os.path.join(PARAMS.modelsDir, "EXP3", "HierarchicalLoc")
    losses = os.listdir(os.path.join(savedModelsDir, "FineLoc"))
    for loss in losses:
        sl = PARAMS.lossAbreviations[losses.index(loss)]

        print(f"\n\nTEST HIERARCHICAL LOCALIZATION, Loss: {sl}\n")

        lossDir_CL = os.path.join(savedModelsDir, "CoarseLoc", sl)
        marginDir_CL = os.path.join(lossDir_CL, os.listdir(lossDir_CL)[0])
        testNetCL = os.path.join(marginDir_CL, os.listdir(marginDir_CL)[0])
        netCL = torch.load(testNetCL)

        lossDir_FL = os.path.join(savedModelsDir, "FineLoc", sl)
        bestErrorRooms = 100*np.ones((len(rooms), 1))
        bestNets = np.zeros((len(rooms), 1))

        for it in range(1, 26):
            netsFL = []
            for room in rooms:
                netsFL.append(torch.load(os.path.join(lossDir_FL, room, "net_it" + str(it))).to(device))
                # netDir = os.path.join(lossDir_FL, room)
                # netsFL.append(torch.load(os.path.join(netDir, os.listdir(netDir)[0])).to(device))

            print(f"It: {it}\n")

            """REPRESENTATIVE IMAGES"""

            descImgRep, descImgRepSunny = [], []
            for i, imgRepData in enumerate(imgRepDataloader, 0):
                imgRep, room = imgRepData[0].to(device), imgRepData[1]
                output = netCL(imgRep).cpu().detach().numpy()[0]
                descImgRep.append(output)
                if not 'SA-A' in room[0]:
                    descImgRepSunny.append(output)
            treeImgRep = KDTree(descImgRep, leaf_size=2)
            treeImgRepSunny = KDTree(descImgRepSunny, leaf_size=2)

            roomAccuracy, envAccuracy = np.zeros(len(condIlum)), np.zeros(len(condIlum))

            """VISUAL MODEL"""

            coordsVMenv, coordsVMrooms, descriptorsVM, treeDescVMrooms = [[], [], []], [], [], []

            for room in rooms:
                descVMroom, coordsVMroom = [], []
                for i, vmData in enumerate(vmDataloader, 0):
                    imgVM, roomVM, coordsImgVM = vmData
                    imgVM = imgVM.to(device)
                    if roomVM[0] == room:
                        output = netsFL[rooms.index(roomVM[0])](imgVM).cpu().detach().numpy()[0]
                        descVMroom.append(output)
                        coordsImgVM = coordsImgVM.detach().numpy()[0]
                        coordsVMroom.append(coordsImgVM)
                        coordsVMenv[get_env(room)].append(coordsImgVM)
                coordsVMrooms.append(coordsVMroom)
                descriptorsVM.append(descVMroom)
                treeDescVMrooms.append(KDTree(descVMroom, leaf_size=2))

            treeCoordsVMenv = []
            for e in range(3):
                treeCoordsVMenv.append(KDTree(coordsVMenv[e], leaf_size=2))


            """TEST"""

            accuracyCoarseLoc, accuracyEnv = np.zeros(len(condIlum) + 1), np.zeros(len(condIlum) + 1)
            recall = np.zeros((len(condIlum) + 1, PARAMS.kMax))

            geomError, minErrorPossible = np.zeros(len(condIlum) + 1), np.zeros(len(condIlum) + 1)
            geomErrorRooms, minErrorRooms = np.zeros((len(condIlum) + 1, len(rooms))), np.zeros((len(condIlum) + 1, len(rooms)))

            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)
                print(f"Test {ilum}\n")
                rooms = sorted(os.listdir(os.path.join(datasetDir, "Test" + ilum)))

                testDataset = exp3_create_datasets.Test(illumination=ilum)
                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                for i, data in enumerate(testDataloader, 0):
                    imgTest, actualRoom, coordsImgTest = data
                    imgTest = imgTest.to(device)

                    """COARSE LOCALIZATION"""

                    output = netCL(imgTest).cpu().detach().numpy()[0]

                    if ilum == "Sunny":
                        repDescriptors = descImgRepSunny[:]
                        tree = treeImgRepSunny
                    else:
                        repDescriptors = descImgRep[:]
                        tree = treeImgRep

                    if sl in ['CL', 'AL']:
                        cosSimilarities = np.dot(repDescriptors, output)
                        idxPredRoom = np.argmax(cosSimilarities)
                    else:
                        _, idxPredRoom = tree.query(output.reshape(1, -1), k=1)
                        idxPredRoom = idxPredRoom[0][0]
                    predictedRoom = rooms[idxPredRoom]
                    predictedEnv, actualEnv = get_env(predictedRoom), get_env(actualRoom[0])

                    if predictedEnv == actualEnv:
                        accuracyEnv[idxIlum] += 1
                        if predictedRoom == actualRoom[0]:
                            accuracyCoarseLoc[idxIlum] += 1

                    """FINE LOCALIZATION"""

                    coordsImgTest = coordsImgTest.detach().numpy()[0]
                    _, idxGeom = treeCoordsVMenv[actualEnv].query(coordsImgTest.reshape(1, -1), k=PARAMS.kMax)
                    idxMinReal = idxGeom[0][0]
                    coordsClosestImg = coordsVMenv[actualEnv][idxMinReal]
                    minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)
                    minErrorRooms[idxIlum][rooms.index(actualRoom[0])] += np.linalg.norm(coordsImgTest - coordsClosestImg)


                    if predictedEnv == actualEnv:

                        output = netsFL[idxPredRoom](imgTest).cpu().detach().numpy()[0]

                        if sl in ['CL', 'AL']:
                            cosSimilarities = np.dot(descriptorsVM[idxPredRoom], output)
                            idxMinPred = np.argmax(cosSimilarities)
                        else:
                            treeDescVM = treeDescVMrooms[idxPredRoom]
                            _, idxDesc = treeDescVM.query(output.reshape(1, -1), k=1)
                            idxMinPred = idxDesc[0][0]

                        coordsPredictedImg  = coordsVMrooms[idxPredRoom][idxMinPred]

                        for k in range(PARAMS.kMax):
                            if coordsPredictedImg[0] == coordsVMenv[actualEnv][idxGeom[0][k]][0]:
                                prediction = True
                                recall[idxIlum][k:] += 1
                                break

                        geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                        geomErrorRooms[idxIlum][rooms.index(actualRoom[0])] += np.linalg.norm(coordsImgTest - coordsPredictedImg)


                for room in range(len(rooms)):
                    auxDir = os.path.join(datasetDir, "Test" + ilum, rooms[room])
                    geomErrorRooms[idxIlum][room] /= len(os.listdir(auxDir))
                    minErrorRooms[idxIlum][room] /= len(os.listdir(auxDir))

                # Results for each lighting condition
                accuracyCoarseLoc[idxIlum] *= 100 / len(testDataloader)
                recall[idxIlum] *= 100 / len(testDataloader)
                geomError[idxIlum] /= accuracyEnv[idxIlum]
                minErrorPossible[idxIlum] /= len(testDataloader)


                print(f"COARSE LOC\nAccuracy: {accuracyCoarseLoc[idxIlum]} %\n")
                print(f"FINE LOC")

                for room in range(len(rooms)):
                    print(f"Error Room {rooms[room]}: {geomErrorRooms[idxIlum][room]} m")
                print(f"Average {ilum} Error: {geomError[idxIlum]} m")
                print("\n")
                writer.writerow([sl, ilum, accuracyCoarseLoc[idxIlum], geomError[idxIlum],
                                 recall[idxIlum][0], recall[idxIlum][round(len(vmDataloader) / 100)]])

            rooms = sorted(os.listdir(os.path.join(datasetDir, "Train")))
            # Average results
            accuracyCoarseLoc[-1] = np.average(accuracyCoarseLoc[0:-1])
            geomError[-1] = np.average(geomError[0:-1])
            minErrorPossible[-1] = np.average(minErrorPossible[0:-1])
            geomErrorRooms[-1] = np.average(geomErrorRooms[0:-1], axis=0)
            minErrorRooms[-1] = np.average(minErrorRooms[0:-1], axis=0)
            recall[-1] = np.average(recall[0:-1], axis=0)

            for room in range(len(rooms)):
                print(f"Average Error Room {rooms[room]}: {geomErrorRooms[-1][room]} m")
            print(f"Average Error: {geomError[-1]} m")
        #     writer.writerow([sl, "Average", accuracyCoarseLoc[-1], geomError[-1],
        #                      recall[-1][0], recall[-1][round(len(vmDataloader)/100)]])
        #
            for room in range(len(rooms)):
                if geomErrorRooms[-1][room] < bestErrorRooms[room]:
                    bestErrorRooms[room] = geomErrorRooms[-1][room]
                    bestNets[room] = it

        for room in rooms:
            print(f"Best net loss {sl}, room {room}: {bestNets[rooms.index(room)]},"
                  f" Geometric Error: {bestErrorRooms[rooms.index(room)]} m")
