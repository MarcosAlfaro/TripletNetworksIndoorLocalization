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
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree
import create_datasets
import create_figures
from config import PARAMETERS

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device0)

device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device1)


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")
figuresDir = os.path.join(baseDir, "FIGURES", "EXP1", "FineLoc")

condIlum = ['Cloudy', 'Night', 'Sunny']
kMax = PARAMETERS.kFineLoc


imgRepDataset = create_datasets.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/")
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

vmDataset = create_datasets.VisualModelTestFineLoc(imageFolderDataset=datasetDir + "/Train/")
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

with open(csvDir + '/Exp1ResultsHierarchicalLoc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    columnsCSV = ["Loss", "Ilum", "CoarseLoc Accuracy"]

    trainDir = os.path.join(datasetDir, "Train")
    trainDataset = dset.ImageFolder(root=trainDir)
    rooms = trainDataset.classes

    for room in rooms:
        columnsCSV.append('Geom Error ' + room)
    columnsCSV.append('Geom Error')
    for k in range(kMax):
        columnsCSV.append('Recall@k' + str(k+1))

    writer.writerow([columnsCSV])
    firstRow = [True, True, True]

    lossesDir_Lf = os.path.join(baseDir, "SAVED_MODELS", "EXP1", "FineLoc")
    losses = os.listdir(lossesDir_Lf)

    for sl in losses:
        # these networks must be renamed and copied in directory "baseDir/SAVED_MODELS/CoarseLoc/BestNets/sl" manually
        idxLoss = PARAMETERS.lossAbreviations.index(sl)
        loss = PARAMETERS.losses[idxLoss]

        lossDir_Lg = os.path.join(baseDir, "SAVED_MODELS", "EXP1", "CoarseLoc", "BestNets", sl)
        testNetLg = os.path.join(lossDir_Lg, os.listdir(lossDir_Lg)[0])
        netLg = torch.load(testNetLg, map_location=device0)

        lossDir_Lf = os.path.join(lossesDir_Lf, sl)
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

            print(f"TEST HIERARCHICAL LOCALIZATION\n Loss: {loss}\n\n")

            """REPRESENTATIVE IMAGES"""

            descImgRep = []
            for i, imgRepData in enumerate(imgRepDataloader, 0):
                imgRep = imgRepData[0].to(device0)
                output = netLg(imgRep).cpu().detach().numpy()[0]
                descImgRep.append(output)
            treeImgRep = KDTree(descImgRep, leaf_size=2)

            """VISUAL MODEL"""

            coordsVM, coordsVMrooms = [], []
            descriptorsVM = []
            treeDescVMrooms = []

            for room in rooms:
                idxRoom = rooms.index(room)
                testNetDir = os.path.join(baseDir, "SAVED_MODELS", "EXP1", "FineLoc", sl, room, testNetsLf[idxRoom])
                netLf = torch.load(testNetDir, map_location=device1)
                descVMroom, coordsVMroom = [], []
                for i, VMdata in enumerate(vmDataloader, 0):

                    imgVM, ind_gt, coordsImgVM = VMdata
                    imgVM = imgVM.to(device1)

                    if ind_gt.detach().numpy()[0] == idxRoom:
                        output = netLf(imgVM).cpu().detach().numpy()[0]
                        descVMroom.append(output)
                        coordsImgVM = coordsImgVM.detach().numpy()[0]
                        coordsVMroom.append(coordsImgVM)
                        coordsVM.append(coordsImgVM)

                coordsVMrooms.append(coordsVMroom)
                descriptorsVM.append(descVMroom)
                treeDescVMrooms.append(KDTree(descVMroom, leaf_size=2))

            treeCoordsVM = KDTree(coordsVM, leaf_size=2)
            """
        
        
        
        
        
        
            """

            recallLF = np.zeros((len(condIlum) + 1, kMax))
            accuracyCoarseLoc = np.zeros(len(condIlum) + 1)
            geomError, minErrorPossible = np.zeros(len(condIlum) + 1), np.zeros(len(condIlum) + 1)
            geomErrorRooms = np.zeros((len(condIlum) + 1, len(rooms)))
            minErrorRooms = np.zeros((len(condIlum) + 1, len(rooms)))
            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)
                print(f"Test {ilum}\n")

                testDataset = create_datasets.TestFineLoc(
                    illumination=ilum, imageFolderDataset=datasetDir + "/Test" + ilum + "/")
                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                coordsMapTest = []

                actualRooms, predRooms = [], []

                for i, data in enumerate(testDataloader, 0):
                    imgTest, actualRoom, coordsImgTest = data
                    imgTestLg = imgTest.to(device0)

                    """COARSE LOCALIZATION"""

                    output = netLg(imgTestLg).cpu().detach().numpy()[0]

                    if sl == 'CL' or sl == 'AL':
                        cosSimilarities = np.dot(descImgRep, output)
                        predictedRoom = np.argmax(cosSimilarities)
                    else:
                        _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                        predictedRoom = predictedRoom[0][0]

                    actualRoom = actualRoom.detach().numpy()[0]
                    if predictedRoom == actualRoom:
                        accuracyCoarseLoc[idxIlum] += 1

                    actualRooms.append(actualRoom)
                    predRooms.append(predictedRoom)

                    """FINE LOCALIZATION"""

                    testNetDir = os.path.join(baseDir, "SAVED_MODELS", "EXP1", "FineLoc", sl, rooms[predictedRoom],
                                              testNetsLf[predictedRoom])
                    netLf = torch.load(testNetDir, map_location=device1)
                    imgTestLf = imgTest.to(device1)
                    output = netLf(imgTestLf).cpu().detach().numpy()[0]

                    if sl == 'CL' or sl == 'AL':
                        cosSimilarities = np.dot(descriptorsVM[predictedRoom], output)
                        idxMinPred = np.argmax(cosSimilarities)
                    else:
                        treeDescVM = treeDescVMrooms[predictedRoom]
                        _, idxDesc = treeDescVM.query(output.reshape(1, -1), k=1)
                        idxMinPred = idxDesc[0][0]

                    coordsImgTest = coordsImgTest.detach().numpy()[0]
                    _, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=kMax)
                    idxMinReal = idxGeom[0][0]

                    coordsPredictedImg = coordsVMrooms[predictedRoom][idxMinPred]
                    coordsClosestImg = coordsVM[idxMinReal]

                    for k in range(kMax):
                        if coordsPredictedImg[0] == coordsVM[idxGeom[0][k]][0]:
                            prediction = True
                            break

                    if actualRoom != predictedRoom:
                        label = "R"
                        if prediction:
                            recallLF[idxIlum][k:] += 1
                    elif prediction:
                        label = str(k+1)
                        recallLF[idxIlum][k:] += 1
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
                    row = ["Loss", "Min Error " + ilum, 100]
                    for room in range(len(rooms)):
                        row.append(minErrorRooms[idxIlum][room])
                    row.append(minErrorPossible[idxIlum])
                    for k in range(kMax):
                        row.append(100)
                    writer.writerow([row])
                    firstRow[idxIlum] = False

                create_figures.display_coord_map(figuresDir, 'Hierarchical',
                                                 coordsVM, coordsMapTest, kMax, ilum, sl)

                create_figures.display_confusion_matrix(actual=actualRooms, predicted=predRooms,
                                                        rooms=rooms, plt_name=os.path.join(figuresDir, 'cm' + sl +
                                                                                           '_' + ilum + '.png'),
                                                        loss=loss, ilum=ilum)

                print(f"COARSE LOC\nAccuracy: {accuracyCoarseLoc[idxIlum]} %\n")
                print(f"FINE LOC")
                print(f"Geometric error: {geomError[idxIlum]} m")
                print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                row = [sl, ilum, accuracyCoarseLoc[idxIlum]]
                for room in range(len(rooms)):
                    row.append(geomErrorRooms[idxIlum][room])
                row.append(geomError[idxIlum])
                for k in range(kMax):
                    print(f"Recall@{k+1}: {recallLF[idxIlum][k]} %")
                    row.append(recallLF[idxIlum][k])
                print("\n")
                writer.writerow([row])

            accuracyCoarseLoc[-1] = np.average(accuracyCoarseLoc[0:-1])
            geomError[-1] = np.average(geomError[0:-1])
            minErrorPossible[-1] = np.average(minErrorPossible[0:-1])
            geomErrorRooms[-1] = np.average(geomErrorRooms[0:-1], axis=0)
            minErrorRooms[-1] = np.average(minErrorRooms[0:-1], axis=0)
            recallLF[-1] = np.average(recallLF[0:-1], axis=0)

            create_figures.error_rooms(figuresDir, geomError, minErrorPossible,
                                       geomErrorRooms, minErrorRooms, rooms, loss)

            row = [sl, "Average", accuracyCoarseLoc[-1]]
            for room in range(len(rooms)):
                row.append(geomErrorRooms[-1][room])
            row.append(geomError[-1])
            for k in range(kMax):
                row.append(recallLF[-1][k])
            writer.writerow([row])

            for room in range(len(rooms)):
                if geomErrorRooms[-1][room] < bestErrorRooms[room]:
                    bestErrorRooms[room] = geomErrorRooms[-1][room]
                    bestNets[room] = testNetsLf[room]

        for room in rooms:
            print(f"Best net loss {loss}, room {room}: {bestNets[rooms.index(room)]},"
                  f" Geometric Error: {bestErrorRooms[rooms.index(room)]} m")
