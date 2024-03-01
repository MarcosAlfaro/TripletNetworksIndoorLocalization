"""
TEST CODE: COARSE LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the room retrieval task

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


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def get_loss(red):
    _, lf = red.split("netLg_")
    lf, _ = lf.split("m")
    return lf


def get_env(predRoom):
    if predRoom <= 8:
        env = 0
    elif 9 <= predRoom <= 16:
        env = 1
    else:
        env = 2
    return env


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS")
figuresDir = create_path(os.path.join(baseDir, "FIGURES", "EXP2", "CoarseLoc"))

condIlum = ['Cloudy', 'Night', 'Sunny']


imgRepDataset = create_datasets2.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/")
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)


with open(csvDir + "/Exp2ResultsCoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Net"]
    for ilum in condIlum:
        rowCSV.extend(["Env " + ilum, 'Room' + ilum])
    rowCSV.extend(["Env Average", "Room Average"])
    writer.writerow(rowCSV)

    netDir = os.path.join(baseDir, "SAVED_MODELS", "EXP2", "CoarseLoc")
    losses = os.listdir(netDir)
    bestAccuracy, bestNet = 0, ""
    for loss in losses:
        if loss == "BestNets":
            continue
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
                print(f"TEST NET {testNet}")
                rowCSV = [testNet]

                """REPRESENTATIVE IMAGES"""

                rooms = dset.ImageFolder(root=datasetDir + "/Train/").classes
                descImgRep, descImgRepSunny = [], []
                for i, imgRepData in enumerate(imgRepDataloader, 0):
                    imgRep = imgRepData[0].to(device)
                    output = net(imgRep).cpu().detach().numpy()[0]
                    descImgRep.append(output)
                    if "SA-A" not in rooms[i]:
                        descImgRepSunny.append(output)
                treeImgRep = KDTree(descImgRep, leaf_size=2)
                treeImgRepSunny = KDTree(descImgRepSunny, leaf_size=2)

                roomAccuracy, envAccuracy = np.zeros(len(condIlum)), np.zeros(len(condIlum))
                actualRooms, predRooms, actualEnvs, predEnvs = [], [], [], []

                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    print(f"Test {ilum}\n")

                    testDatasetDir = datasetDir + "/Test" + ilum + "/"
                    testDataset = create_datasets2.TestCoarseLoc(illumination=ilum, imageFolderDataset=testDatasetDir)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    ilumDataset = dset.ImageFolder(root=datasetDir + "/Test" + ilum + "/")
                    roomsIlum = ilumDataset.classes

                    actualRoomsIlum, predRoomsIlum, actualEnvsIlum, predEnvsIlum = [], [], [], []

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, actualEnv, actualRoom = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]

                        actualRoom = actualRoom.detach().numpy()[0]
                        actualEnv = actualEnv.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':
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

                        actualEnvsIlum.append(actualEnv)
                        predEnvsIlum.append(predictedEnv)
                        actualEnvs.append(actualEnv)
                        predEnvs.append(predictedEnv)

                        actualRoomsIlum.append(actualRoom)
                        predRoomsIlum.append(predictedRoom)
                        actualRooms.append(actualRoom)
                        predRooms.append(predictedRoom)

                        if predictedEnv == actualEnv:
                            envAccuracy[idxIlum] += 1
                            if predictedRoom == actualRoom:
                                roomAccuracy[idxIlum] += 1

                    roomAccuracy[idxIlum] *= 100 / len(testDataloader)
                    envAccuracy[idxIlum] *= 100 / len(testDataloader)

                    create_figures2.display_confusion_matrix_env(
                        actual=actualEnvsIlum, predicted=predEnvsIlum,
                        plt_name=os.path.join(figuresDir, "exp2" + "cmEnv_" + loss + "_" + ilum + '.png'),
                        rooms=rooms, loss=loss, ilum=ilum)

                    create_figures2.display_confusion_matrix_room(
                        actual=actualRoomsIlum, predicted=predRoomsIlum,
                        plt_name=os.path.join(figuresDir, "exp2" + "cmRoom_" + loss + "_" + ilum + '.png'),
                        rooms=roomsIlum, loss=loss, ilum=ilum)

                    print(f'{ilum} accuracy:')
                    print(f'Environment: {envAccuracy[idxIlum]} %')
                    print(f'Room: {roomAccuracy[idxIlum]} %')

                    rowCSV.extend([envAccuracy[idxIlum], roomAccuracy[idxIlum]])

                create_figures2.display_confusion_matrix_env(
                    actual=actualEnvs, predicted=predEnvs,
                    plt_name=os.path.join(figuresDir, "exp2" + "cmEnv_" + loss + "_Avg.png"),
                    rooms=rooms, loss=loss, ilum="Average")

                create_figures2.display_confusion_matrix_room(
                    actual=actualRooms, predicted=predRooms,
                    plt_name=os.path.join(figuresDir, "exp2cmRoom_" + loss + "_Avg.png"),
                    rooms=rooms, loss=loss, ilum="Average")

                avgEnvAccuracy, avgRoomAccuracy = np.average(envAccuracy), np.average(roomAccuracy)

                if avgRoomAccuracy > bestMarginAccuracy:
                    bestMarginNet = testNet
                    bestMarginAccuracy = avgRoomAccuracy
                    if avgRoomAccuracy > bestLossAccuracy:
                        bestLossNet = testNet
                        bestLossAccuracy = avgRoomAccuracy
                        if avgRoomAccuracy > bestAccuracy:
                            bestNet = testNet
                            bestAccuracy = avgRoomAccuracy

                print(f'Average accuracy:')
                print(f'Environment: {avgEnvAccuracy} %\n')
                print(f'Room: {avgRoomAccuracy} %\n')

                rowCSV.extend([avgEnvAccuracy, avgRoomAccuracy])
                writer.writerow(rowCSV)

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Accuracy: {bestMarginAccuracy} %")
        if bestLossNet != "":
            print(f"Best net loss {loss}: {bestLossNet}, Accuracy: {bestLossAccuracy} %")
    if bestNet != "":
        print(f"Best net: {bestNet}, Accuracy: {bestAccuracy} %")
