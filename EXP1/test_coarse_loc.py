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
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree
import torch.nn as nn
from torchvision.models import VGG16_Weights

import create_datasets
import create_figures


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def get_loss(red):
    _, lf = red.split("netLg_")
    lf, _ = lf.split("m")
    return lf


class TripletNetwork(nn.Module):

    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn1 = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * 512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def forward_once(self, x):
        verbose = False

        if verbose:
            print("Input: ", x.size())

        out = self.cnn1(x)

        if verbose:
            print("Output matricial: ", out.size())

        out = self.avgpool(out)
        if verbose:
            print("Output avgpool: ", out.size())
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        norm = True
        if norm:
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out

    def forward(self, input1):
        out1 = self.forward_once(input1)
        return out1


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
figuresDir = os.path.join(baseDir, "FIGURES", "CoarseLoc")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")


imgRepDataset = create_datasets.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/",
                                                     transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                   transforms.ToTensor()
                                                                                   ]),
                                                     should_invert=False)

imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

cloudyDataset = dset.ImageFolder(root=datasetDir + "/TestCloudy/")
rooms = cloudyDataset.classes

vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)

with open(csvDir + "/ResultsCoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Net", "Cloudy", "Night", "Sunny", "Average"])

    netDir = os.path.join(baseDir, "SAVED_MODELS", "CoarseLoc")
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
                print(f"TEST NET {testNet}")

                """REPRESENTATIVE IMAGES"""

                descImgRep = []
                for i, imgRepData in enumerate(imgRepDataloader, 0):
                    imgRep, _ = imgRepData
                    imgRep = imgRep.to(device)

                    output = net(imgRep)
                    output = output.cpu()

                    descImgRep.append(output.detach().numpy()[0])

                treeImgRep = KDTree(descImgRep, leaf_size=2)

                condIlum = ['Cloudy', 'Night', 'Sunny']
                accuracy = []

                for ilum in condIlum:

                    print(f"Test {ilum}\n")

                    testDatasetDir = datasetDir + "/Test" + ilum + "/"

                    testDataset = create_datasets.TestCoarseLoc(illumination=ilum, imageFolderDataset=testDatasetDir,
                                                                transform=transforms.Compose([
                                                                          transforms.Resize((128, 512)),
                                                                          transforms.ToTensor()]), should_invert=False)

                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    correct = 0
                    actualRooms, predRooms = [], []

                    for i, data in enumerate(testDataloader, 0):

                        imgTest, actualRoom = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest)
                        output = output.cpu()
                        output = output.detach().numpy()[0]

                        actualRoom = actualRoom.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':

                            cosMax = 0
                            for desc in descImgRep:
                                cosSimilarity = np.dot(desc, output)
                                if cosSimilarity > cosMax:
                                    cosMax = cosSimilarity
                                    predictedRoom = descImgRep.index(desc)
                        else:
                            _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                            predictedRoom = predictedRoom[0][0]

                        actualRooms.append(actualRoom)
                        predRooms.append(predictedRoom)

                        if predictedRoom == actualRoom:
                            correct += 1

                    create_figures.display_confusion_matrix(actual=actualRooms, predicted=predRooms,
                                                            plt_name=os.path.join(figuresDir, "cm_" + loss +
                                                                                  "_" + ilum + '.png'),
                                                            rooms=rooms, loss=loss, ilum=ilum)

                    accuracy.append(100 * correct / len(testDataloader))
                avgAccuracy = (accuracy[0] + accuracy[1] + accuracy[2]) / 3
                if avgAccuracy > bestMarginAccuracy:
                    bestMarginNet = testNet
                    bestMarginAccuracy = avgAccuracy
                if avgAccuracy > bestLossAccuracy:
                    bestLossNet = testNet
                    bestLossAccuracy = avgAccuracy
                if avgAccuracy > bestAccuracy:
                    bestNet = testNet
                    bestAccuracy = avgAccuracy

                print(f'Cloudy accuracy: {accuracy[0]}%')
                print(f'Night accuracy: {accuracy[1]}%')
                print(f'Sunny accuracy: {accuracy[2]}%')
                print(f'Average accuracy: {avgAccuracy}%\n')

                writer.writerow([testNet, accuracy[0], accuracy[1], accuracy[2], avgAccuracy])

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Accuracy: {bestMarginAccuracy} %")
        if bestLossNet != "":
            print(f"Best net loss {loss}: {bestLossNet}, Accuracy: {bestLossAccuracy} %")
    if bestNet != "":
        print(f"Best net: {bestNet}, Accuracy: {bestAccuracy} %")
