"""
TEST CODE: COARSE LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the room retrieval task


Test dataset:
Cloudy: seq2cloudy2 (2595 images)
Night: seq2night2 (2707 images)
Sunny: seq2sunny2 (2114 images)

-each test image is compared with the representative image of every room
-the closest representative descriptor indicates the retrieved room

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


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")
figuresDir = create_path(os.path.join(baseDir, "FIGURES", "EXP1", "CoarseLoc"))


imgRepDataset = create_datasets.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/")
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

cloudyDataset = dset.ImageFolder(root=datasetDir + "/TestCloudy/")
rooms = cloudyDataset.classes

condIlum = ['Cloudy', 'Night', 'Sunny']


with open(csvDir + "/Exp1ResultsCoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Net"]
    for ilum in condIlum:
        rowCSV.append(ilum)
    rowCSV.append("Average")
    writer.writerow(rowCSV)

    netDir = os.path.join(baseDir, "SAVED_MODELS", "EXP1", "CoarseLoc")
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

                    testDatasetDir = datasetDir + "/Test" + ilum + "/"
                    testDataset = create_datasets.TestCoarseLoc(illumination=ilum, imageFolderDataset=testDatasetDir)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    actualRooms, predRooms = [], []
                    for i, data in enumerate(testDataloader, 0):

                        imgTest, actualRoom = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]
                        actualRoom = actualRoom.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':
                            cosSimilarities = np.dot(descImgRep, output)
                            predictedRoom = np.argmax(cosSimilarities)
                        else:
                            _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                            predictedRoom = predictedRoom[0][0]

                        actualRooms.append(actualRoom)
                        predRooms.append(predictedRoom)

                        if predictedRoom == actualRoom:
                            accuracy[idxIlum] += 1

                    create_figures.display_confusion_matrix(actual=actualRooms, predicted=predRooms,
                                                            plt_name=os.path.join(figuresDir, "cm_" + loss +
                                                                                  "_" + ilum + '.png'),
                                                            rooms=rooms, loss=loss, ilum=ilum)

                    accuracy[idxIlum] *= 100 / len(testDataloader)
                    print(f'{ilum} accuracy: {accuracy[idxIlum]} %')
                    rowCSV.append(accuracy[idxIlum])

                avgAccuracy = np.average(accuracy)

                if avgAccuracy > bestMarginAccuracy:
                    bestMarginNet = testNet
                    bestMarginAccuracy = avgAccuracy
                    if avgAccuracy > bestLossAccuracy:
                        bestLossNet = testNet
                        bestLossAccuracy = avgAccuracy
                        if avgAccuracy > bestAccuracy:
                            bestNet = testNet
                            bestAccuracy = avgAccuracy

                print(f'Average accuracy: {avgAccuracy}%\n')
                rowCSV.append(avgAccuracy)
                writer.writerow(rowCSV)

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Accuracy: {bestMarginAccuracy} %")
        if bestLossNet != "":
            print(f"Best net loss {loss}: {bestLossNet}, Accuracy: {bestLossAccuracy} %")
    if bestNet != "":
        print(f"Best net: {bestNet}, Accuracy: {bestAccuracy} %")
