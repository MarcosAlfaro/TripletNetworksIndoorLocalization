"""
TEST CODE: HIERARCHICAL LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the hierarchical method

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
    -the nearest neighbour indicates the retrieved coordinates
"""

import os
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree

import create_datasets2
import create_figures2
from config import PARAMETERS


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def get_env(predRoom):
    if predRoom <= 8:
        env = 0
    elif 9 <= predRoom <= 16:
        env = 1
    else:
        env = 2
    return env


device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device0)

device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device1)


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS")
figuresDir = create_path(os.path.join(baseDir, "FIGURES", "EXP2", "FineLoc"))

condIlum = ['Cloudy', 'Night', 'Sunny']
# condIlum = ['Sunny']
kMax = PARAMETERS.kFineLoc


imgRepDataset = create_datasets2.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/")
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

vmDataset = create_datasets2.VisualModelTestFineLoc(imageFolderDataset=datasetDir + "/Train/")
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

with open(csvDir + '/Exp2ResultsHierarchicalLoc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    columnsCSV = ["Loss", "Ilum", "CoarseLoc Accuracy"]

    trainDir = os.path.join(datasetDir, "Train")
    trainDataset = dset.ImageFolder(root=trainDir)
    rooms = trainDataset.classes

    for room in rooms:
        columnsCSV.append('Geom Error ' + room)
    columnsCSV.append('Geom Error')
    for k in range(kMax):
        columnsCSV.append('Recallk' + str(k + 1))

    writer.writerow([columnsCSV])
    firstRow = [True, True, True]

    lossesDir_Lf = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "FineLoc")
    losses = os.listdir(lossesDir_Lf)

    for sl in losses:
        # these networks must be renamed and copied in directory "baseDir/SAVED_MODELS/CoarseLoc/BestNets/sl" manually
        idxLoss = PARAMETERS.lossAbreviations.index(sl)
        loss = PARAMETERS.losses[idxLoss]

        lossDir_Lg = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "CoarseLoc", "BestNets", sl)
        testNetLg = os.path.join(lossDir_Lg, os.listdir(lossDir_Lg)[0])
        netLg = torch.load(testNetLg, map_location=device0)

        lossDir_Lf = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "FineLoc", sl)
        allNetsLf = []
        numMaxNets = 0
        bestNets = []
        for room in rooms:
            bestNets.append("")
            allNetsLf.append([])
            roomDir = os.path.join(lossDir_Lf, room)
            allNetsLf[rooms.index(room)].extend(os.listdir(roomDir))
            if len(os.listdir(roomDir)) > numMaxNets:
                numMaxNets = len(os.listdir(roomDir))

        bestErrorRooms = 100 * np.ones(len(rooms))
        for n in range(numMaxNets):
            testNetsLf = []
            for room in rooms:
                if n < len(allNetsLf[rooms.index(room)]):
                    testNetsLf.append(allNetsLf[rooms.index(room)][n])
                else:
                    testNetsLf.append(allNetsLf[rooms.index(room)][-1])

            netLg = torch.load(testNetLg).to(device0)

            print(f"\n\nTEST HIERARCHICAL LOCALIZATION\n Loss: {loss}\n")

            """REPRESENTATIVE IMAGES"""

            descImgRep, descImgRepSunny = [], []
            for i, imgRepData in enumerate(imgRepDataloader, 0):
                imgRep = imgRepData[0].to(device0)
                output = netLg(imgRep).cpu().detach().numpy()[0]
                descImgRep.append(output)
                if "SA-A" not in rooms[i]:
                    descImgRepSunny.append(output)
            treeImgRep = KDTree(descImgRep, leaf_size=2)
            treeImgRepSunny = KDTree(descImgRepSunny, leaf_size=2)

            """VISUAL MODEL"""
            descriptorsVM = []
            treeDescVMrooms = []
            coordsVMrooms = []
            coordsVMenv = [[], [], []]
            for room in rooms:
                idxRoom = rooms.index(room)
                testNetDir = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "FineLoc", sl, room, testNetsLf[idxRoom])
                netLf = torch.load(testNetDir)

                descVMroom, coordsVMroom = [], []

                for i, vmData in enumerate(vmDataloader, 0):
                    imgVM, idxEnvVM, idxRoomVM, coordsImgVM = vmData
                    imgVM = imgVM.to(device1)

                    idxRoomVM = idxRoomVM.detach().numpy()[0]
                    if idxRoomVM == idxRoom:
                        output = netLf(imgVM).cpu().detach().numpy()[0]
                        coordsImgVM = coordsImgVM.detach().numpy()[0]
                        idxEnvVM = idxEnvVM.detach().numpy()[0]

                        descVMroom.append(output)
                        coordsVMroom.append(coordsImgVM)
                        coordsVMenv[idxEnvVM].append(coordsImgVM)

                coordsVMrooms.append(coordsVMroom)
                descriptorsVM.append(descVMroom)
                treeDescVMrooms.append(KDTree(descVMroom, leaf_size=2))

            treeCoordsVMenv = []
            for e in range(3):
                treeCoordsVMenv.append(KDTree(coordsVMenv[e], leaf_size=2))
            """
        
        
        
        
        
        
            """

            actualRooms, predRooms, actualEnvs, predEnvs = [], [], [], []
            accuracyRoom, accuracyEnv = np.zeros(len(condIlum)), np.zeros(len(condIlum))
            accuracyEnvRooms = np.zeros((len(condIlum), len(rooms)))

            recallLF = np.zeros((len(condIlum), kMax))
            geomError, minErrorPossible = np.zeros(len(condIlum)), np.zeros(len(condIlum))
            geomErrorRooms, minErrorRooms = np.zeros((len(condIlum), len(rooms))), np.zeros((len(condIlum), len(rooms)))

            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)

                print(f"Test {ilum}\n")

                testDataset = create_datasets2.TestFineLoc(
                    illumination=ilum, imageFolderDataset=datasetDir + "/Test" + ilum + "/")
                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                ilumDataset = dset.ImageFolder(root=datasetDir + "/Test" + ilum + "/")
                roomsIlum = ilumDataset.classes

                actualRoomsIlum, predRoomsIlum, actualEnvsIlum, predEnvsIlum = [], [], [], []

                for i, data in enumerate(testDataloader, 0):
                    imgTest, actualEnv, actualRoom, coordsImgTest = data
                    imgTestLg = imgTest.to(device0)

                    """COARSE LOCALIZATION"""

                    actualRoom = actualRoom.detach().numpy()[0]
                    actualEnv = actualEnv.detach().numpy()[0]

                    output = netLg(imgTestLg).cpu().detach().numpy()[0]

                    if sl == 'CL' or sl == 'AL':
                        if ilum == "Sunny":
                            repDescriptors = descImgRepSunny[:]
                        else:
                            repDescriptors = descImgRep[:]
                        cosSimilarities = np.dot(repDescriptors, output)
                        predictedRoom = np.argmax(cosSimilarities)
                    else:
                        if ilum == "Sunny":
                            _, predictedRoom = treeImgRepSunny.query(output.reshape(1, -1), k=1)
                        else:
                            _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                        predictedRoom = predictedRoom[0][0]

                    if ilum == "Sunny" and "SA-B" in roomsIlum[predictedRoom]:
                        predictedRoom += 8

                    predictedEnv = get_env(predictedRoom)

                    if predictedEnv == actualEnv:
                        accuracyEnv[idxIlum] += 1
                        accuracyEnvRooms[idxIlum][actualRoom] += 1
                        if predictedRoom == actualRoom:
                            accuracyRoom[idxIlum] += 1

                    actualEnvsIlum.append(actualEnv)
                    predEnvsIlum.append(predictedEnv)
                    actualEnvs.append(actualEnv)
                    predEnvs.append(predictedEnv)

                    actualRoomsIlum.append(actualRoom)
                    predRoomsIlum.append(predictedRoom)
                    actualRooms.append(actualRoom)
                    predRooms.append(predictedRoom)

                    """FINE LOCALIZATION"""
                    testNetDir = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "FineLoc",
                                              sl, rooms[predictedRoom], testNetsLf[predictedRoom])
                    netLf = torch.load(testNetDir, map_location=device1)
                    imgTestLf = imgTest.to(device1)
                    output = netLf(imgTestLf).cpu().detach().numpy()[0]
                    coordsImgTest = coordsImgTest.detach().numpy()[0]

                    if sl == 'CL' or sl == 'AL':
                        cosSimilarities = np.dot(descriptorsVM[predictedRoom], output)
                        idxMinPred = np.argmax(cosSimilarities)
                    else:
                        treeDescVM = treeDescVMrooms[predictedRoom]
                        _, idxDesc = treeDescVM.query(output.reshape(1, -1), k=1)
                        idxMinPred = idxDesc[0][0]

                    _, idxGeom = treeCoordsVMenv[actualEnv].query(coordsImgTest.reshape(1, -1), k=kMax)
                    idxMinReal = idxGeom[0][0]

                    coordsPredictedImg = coordsVMrooms[predictedRoom][idxMinPred]
                    coordsClosestImg = coordsVMenv[actualEnv][idxMinReal]

                    for k in range(kMax):
                        if coordsPredictedImg[0] == coordsVMenv[actualEnv][idxGeom[0][k]][0]:
                            prediction = True
                            break

                    if prediction:
                        recallLF[idxIlum][k:] += 1

                    if predictedEnv == actualEnv:
                        geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                        # if 'SA-B' in roomsIlum[actualRoom] and ilum == "Sunny":
                        #     geomErrorRooms[idxIlum][actualRoom+8] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                        # else:
                        geomErrorRooms[idxIlum][actualRoom] += np.linalg.norm(coordsImgTest - coordsPredictedImg)

                    minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)
                    # if 'SA-B' in roomsIlum[actualRoom] and ilum == "Sunny":
                    #     minErrorRooms[idxIlum][actualRoom + 8] += np.linalg.norm(coordsImgTest - coordsClosestImg)
                    # else:
                    minErrorRooms[idxIlum][actualRoom] += np.linalg.norm(coordsImgTest - coordsClosestImg)

                for room in range(len(rooms)):
                    testRoomDir = os.path.join(datasetDir, "Test" + ilum, rooms[room])
                    if room < 9 or room > 16 or ilum != "Sunny":
                        geomErrorRooms[idxIlum][room] /= accuracyEnvRooms[idxIlum][room]
                        minErrorRooms[idxIlum][room] /= len(os.listdir(testRoomDir))

                accuracyRoom[idxIlum] *= 100 / len(testDataloader)
                recallLF[idxIlum] *= 100 / len(testDataloader)
                geomError[idxIlum] /= accuracyEnv[idxIlum]
                minErrorPossible[idxIlum] /= len(testDataloader)
                if firstRow[idxIlum]:
                    row = ["Loss", "Min Error " + ilum, 100]
                    for room in range(len(rooms)):
                        row.append(minErrorRooms[idxIlum][room])
                    row.append(minErrorPossible[idxIlum])
                    for k in range(kMax):
                        row.append(100)
                    writer.writerow([row])
                    firstRow[idxIlum] = False

                create_figures2.display_confusion_matrix_room(
                    actual=actualRooms, predicted=predRooms,
                    rooms=rooms, plt_name=os.path.join(figuresDir, 'cm' + sl + '_' + ilum + '.png'),
                    loss=loss, ilum=ilum)

                print(f"COARSE LOC\nAccuracy: {accuracyRoom[idxIlum]} %\n")
                print(f"FINE LOC")

                print(f"Geometric error: {geomError[idxIlum]} m")
                print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                row = [sl, ilum, accuracyRoom[idxIlum]]
                for room in range(len(rooms)):
                    row.append(geomErrorRooms[idxIlum][room])
                row.append(geomError[idxIlum])
                for k in range(kMax):
                    print(f"Recall@{k+1}: {recallLF[idxIlum][k]} %")
                    row.append(recallLF[idxIlum][k])
                print("\n")
                writer.writerow([row])

            avgAccuracyRoom = np.average(accuracyRoom)
            avgGeomError = np.average(geomError)
            avgRecallLF = np.average(recallLF, axis=0)
            avgGeomErrorRooms = np.average(geomErrorRooms, axis=0)
            avgMinErrorRooms = np.average(minErrorRooms, axis=0)
            avgGeomErrorRooms[8:15] *= 1.5
            avgMinErrorRooms[8:15] *= 1.5

            create_figures2.error_rooms(figuresDir, geomError, minErrorPossible,
                                        geomErrorRooms, minErrorRooms, rooms, loss)

            row = [sl, "Average", avgAccuracyRoom]
            for room in range(len(rooms)):
                row.append(avgGeomErrorRooms[room])
            row.append(avgGeomError)
            for k in range(kMax):
                row.append(avgRecallLF[k])
            writer.writerow([row])

            for room in range(len(rooms)):
                if avgGeomErrorRooms[room] < bestErrorRooms[room]:
                    bestErrorRooms[room] = avgGeomErrorRooms[room]
                    bestNets[room] = testNetsLf[room]

        for room in rooms:
            print(f"Best net loss {loss}, room {room}: {bestNets[rooms.index(room)]},"
                  f" Geometric Error: {bestErrorRooms[rooms.index(room)]} m")
