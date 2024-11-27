"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to test triplet networks to perform the coarse localization (second stage of hierarchical loc.)

Test dataset:
Cloudy: seq2cloudy2 (2595 images), Night: seq2night2 (2707 images), Sunny: seq2sunny2 (2114 images)

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in two steps:

-Coarse step: room retrieval task
    -each test image is compared with the representative image of every room
    -the closest representative descriptor indicates the retrieved room
    -metric: room retrieval accuracy (%)

-Fine step: obtain the coordinates of the robot inside the retrieved room:
    -each test image is compared with the images of the visual model of the retrieved room
    -the nearest neighbor indicates the retrieved coordinates
    -metric: geometric error (m)

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
import exp1_create_datasets
import create_figures
from functions import create_path
from config import PARAMS

device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


csvDir = os.path.join(PARAMS.csvDir, "RESULTS")
figuresDir = create_path(os.path.join(PARAMS.figuresDir, "EXP1", "HierarchicalLoc"))


datasetDir = os.path.join(PARAMS.datasetDir, "FRIBURGO_A")
trainDir = os.path.join(datasetDir, "Train")
trainDataset = dset.ImageFolder(root=trainDir)
rooms = trainDataset.classes

condIlum = ['Cloudy', 'Night', 'Sunny']


imgRepDataset = exp1_create_datasets.RepImages()
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

vmDataset = exp1_create_datasets.VisualModel()
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

with open(csvDir + '/Exp1HierarchicalLoc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Ilum", "CoarseLoc Accuracy", "Geom Error", "Recall@1", "Recall@1%"])

    savedModelsDir = os.path.join(PARAMS.modelsDir, "EXP1", "HierarchicalLoc")
    losses = PARAMS.selectedLosses
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

            descImgRep = []
            for i, imgRepData in enumerate(imgRepDataloader, 0):
                imgRep = imgRepData[0].to(device)
                output = netCL(imgRep).cpu().detach().numpy()[0]
                descImgRep.append(output)
            treeImgRep = KDTree(descImgRep, leaf_size=2)

            """VISUAL MODEL"""

            coordsVM, coordsVMrooms, descriptorsVM, treeDescVMrooms = [], [], [], []

            for room in rooms:
                idxRoom = rooms.index(room)
                descVMroom, coordsVMroom = [], []
                for i, vmData in enumerate(vmDataloader, 0):
                    imgVM, ind_gt, coordsImgVM = vmData
                    imgVM = imgVM.to(device)
                    if ind_gt.detach().numpy()[0] == idxRoom:
                        output = netsFL[idxRoom](imgVM).cpu().detach().numpy()[0]
                        descVMroom.append(output)
                        coordsImgVM = coordsImgVM.detach().numpy()[0]
                        coordsVMroom.append(coordsImgVM)
                        coordsVM.append(coordsImgVM)
                coordsVMrooms.append(coordsVMroom)
                descriptorsVM.append(descVMroom)
                treeDescVMrooms.append(KDTree(descVMroom, leaf_size=2))

            treeCoordsVM = KDTree(coordsVM, leaf_size=2)


            """TEST"""

            accuracyCoarseLoc = np.zeros(len(condIlum) + 1)
            recall = np.zeros((len(condIlum) + 1, PARAMS.kMax))

            geomError, minErrorPossible = np.zeros(len(condIlum) + 1), np.zeros(len(condIlum) + 1)
            geomErrorRooms, minErrorRooms = np.zeros((len(condIlum) + 1, len(rooms))), np.zeros((len(condIlum) + 1, len(rooms)))

            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)
                print(f"Test {ilum}\n")

                testDataset = exp1_create_datasets.Test(illumination=ilum)
                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                coordsMapTest = []

                for i, data in enumerate(testDataloader, 0):
                    imgTest, actualRoom, coordsImgTest = data
                    imgTest = imgTest.to(device)

                    """COARSE LOCALIZATION"""

                    output = netCL(imgTest).cpu().detach().numpy()[0]

                    if sl in ['CL', 'AL']:
                        cosSimilarities = np.dot(descImgRep, output)
                        predictedRoom = np.argmax(cosSimilarities)
                    else:
                        distances, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=9)
                        predictedRoom = predictedRoom[0][0]

                    actualRoom = actualRoom.detach().numpy()[0]
                    if predictedRoom == actualRoom:
                        accuracyCoarseLoc[idxIlum] += 1

                    """FINE LOCALIZATION"""
                    output = netsFL[predictedRoom](imgTest).cpu().detach().numpy()[0]

                    if sl in ['CL', 'AL']:
                        cosSimilarities = np.dot(descriptorsVM[predictedRoom], output)
                        idxMinPred = np.argmax(cosSimilarities)
                    else:
                        treeDescVM = treeDescVMrooms[predictedRoom]
                        _, idxDesc = treeDescVM.query(output.reshape(1, -1), k=1)
                        idxMinPred = idxDesc[0][0]

                    coordsImgTest = coordsImgTest.detach().numpy()[0]
                    _, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=PARAMS.kMax)
                    idxMinReal = idxGeom[0][0]

                    coordsPredictedImg, coordsClosestImg = coordsVMrooms[predictedRoom][idxMinPred], coordsVM[idxMinReal]

                    for k in range(PARAMS.kMax):
                        if coordsPredictedImg[0] == coordsVM[idxGeom[0][k]][0]:
                            prediction = True
                            break

                    if actualRoom != predictedRoom:
                        label = "R"
                        if prediction:
                            recall[idxIlum][k:] += 1
                    elif prediction:
                        label = str(k+1)
                        recall[idxIlum][k:] += 1
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

                # Results for each lighting condition
                accuracyCoarseLoc[idxIlum] *= 100 / len(testDataloader)
                recall[idxIlum] *= 100 / len(testDataloader)
                geomError[idxIlum] /= len(testDataloader)
                minErrorPossible[idxIlum] /= len(testDataloader)

                create_figures.display_coord_map(figuresDir, 'Hierarchical',
                                                 coordsVM, coordsMapTest, PARAMS.kMax, ilum, sl)

                print(f"COARSE LOC\nAccuracy: {accuracyCoarseLoc[idxIlum]} %\n")
                print(f"FINE LOC")

                for room in range(len(rooms)):
                    print(f"Error Room {rooms[room]}: {geomErrorRooms[idxIlum][room]} m")
                print(f"Average {ilum} Error: {geomError[idxIlum]} m")
                print("\n")
                writer.writerow([sl, ilum, accuracyCoarseLoc[idxIlum], geomError[idxIlum],
                                 recall[idxIlum][0], recall[idxIlum][round(len(vmDataloader) / 100)]])

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
            writer.writerow([sl, "Average", accuracyCoarseLoc[-1], geomError[-1],
                             recall[-1][0], recall[-1][round(len(vmDataloader)/100)]])

            for room in range(len(rooms)):
                if geomErrorRooms[-1][room] < bestErrorRooms[room]:
                    bestErrorRooms[room] = geomErrorRooms[-1][room]
                    bestNets[room] = it

        for room in rooms:
            print(f"Best net loss {sl}, room {room}: {bestNets[rooms.index(room)]},"
                  f" Geometric Error: {bestErrorRooms[rooms.index(room)]} m")
