"""
TRAIN CODE: FINE LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the fine step

Model:  network saved in the coarse step
Architecture: Triplet Network
A network is trained for each room

Dataset: COLD-Freiburg, Seq2Cloudy3 Sampled (1 out of 5)
556 cloudy images
Training samples: random choice
Anchor, positive, negative -> same room
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos < rNeg


Validation dataset: 556 cloudy images Seq2Cloudy3 Sampled (1 out of 5)
* The images are different from the training set
-each test image is compared with the images of the visual model of the room
-the nearest neighbour indicates the retrieved coordinates

Visual model dataset: the training set is employed as visual model
"""


import csv
import os
import numpy as np
from sklearn.neighbors import KDTree
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision.models import VGG16_Weights

import create_datasets2
import losses
from config import PARAMETERS

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")

trainingDir = os.path.join(datasetDir, "Entrenamiento")
trainingDataset = dset.ImageFolder(root=trainingDir)
rooms = trainingDataset.classes


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


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    run()

vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)


with open(csvDir + "/TrainingFineLoc" + str(PARAMETERS.numExp) + ".csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Loss", "Margin", "Room", "Epoch", "It", "Validation accuracy", "Geometric error"])

    selectedLosses = PARAMETERS.lossesFineLocTraining

    for lossFunction in selectedLosses:
        if lossFunction == 'triplet loss':
            criterion = losses.TripletLoss()
        elif lossFunction == 'lifted embedding':
            criterion = losses.LiftedEmbeddingLoss()
        elif lossFunction == 'lazy triplet':
            criterion = losses.LazyTripletLoss()
        elif lossFunction == 'semi hard':
            criterion = losses.SemiHardLoss()
        elif lossFunction == 'batch hard':
            criterion = losses.BatchHardLoss()
        elif lossFunction == 'circle loss':
            criterion = losses.CircleLoss()
        elif lossFunction == 'angular loss':
            criterion = losses.AngularLoss()
        else:
            continue

        idxLoss = PARAMETERS.losses.index(lossFunction)
        sl = PARAMETERS.lossAbreviations[idxLoss]
        margin = PARAMETERS.marginsFineLoc[idxLoss]
        trainNet = os.path.join(baseDir, "SAVED_MODELS", "CoarseLoc", sl,
                                PARAMETERS.trainedNetsCoarseLoc[idxLoss])

        for rNeg in PARAMETERS.rNegFineLoc:

            for room in rooms:

                if not os.path.exists(os.path.join(baseDir, "SAVED_MODELS", "FineLoc", sl, room)):
                    os.mkdir(os.path.join(baseDir, "SAVED_MODELS", "FineLoc", sl, room))
                netDir = os.path.join(baseDir, "SAVED_MODELS", "FineLoc", sl, room)

                visualModelDataset = create_datasets2.VisualModelTrainFineLoc(
                    currentRoom=room, imageFolderDataset=datasetDir + "/Entrenamiento/" + room,
                    transform=transforms.Compose([transforms.Resize((128, 512)),
                                                  transforms.ToTensor()
                                                  ]),
                    should_invert=False)
                visualModelDataloader = DataLoader(visualModelDataset, shuffle=False, num_workers=0, batch_size=1)

                coordsVM = []
                for i, visualModelData in enumerate(visualModelDataloader, 0):
                    _, coords = visualModelData
                    coordsVM.append(coords.detach().numpy()[0])
                treeCoordsVM = KDTree(coordsVM, leaf_size=2)

                trainingDataset = create_datasets2.TrainFineLoc(
                    rNeg=rNeg, currentRoom=room, imageFolderDataset=dset.ImageFolder(datasetDir + "/Entrenamiento/"),
                    transform=transforms.Compose([transforms.Resize((128, 512)),
                                                  transforms.ToTensor()
                                                  ]),
                    should_invert=False)

                trainDataloader = DataLoader(trainingDataset, shuffle=True, num_workers=0,
                                             batch_size=16)

                validationDataset = create_datasets2.ValidationFineLoc(
                    currentRoom=room, imageFolderDataset=datasetDir + "/Validation/" + room,
                    transform=transforms.Compose([transforms.Resize((128, 512)),
                                                  transforms.ToTensor()
                                                  ]),
                    should_invert=False)

                validationDataloader = DataLoader(validationDataset, shuffle=False, num_workers=0, batch_size=1)

                net = torch.load(trainNet).to(device)

                optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

                print("\nNew training: ")
                print(f"Network model: {trainNet}")
                print(f"Loss: {lossFunction}, margin/alpha: {margin}")
                print(f"Room: {room}\n\n")

                bestError, maxAccuracy = 1000, 0
                i, j = 0, 0

                for epoch in range(0, PARAMETERS.numEpochsFineLoc):
                    print(f"Epoch {epoch}\n")
                    for i, data in enumerate(trainDataloader, 0):

                        anc, pos, neg = data
                        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

                        optimizer.zero_grad()

                        output1, output2, output3 = net(anc), net(pos), net(neg)

                        loss = criterion(output1, output2, output3, margin)
                        loss.backward()

                        optimizer.step()

                        if i % PARAMETERS.showLoss == 0:
                            print(f"Epoch {epoch}, It {i}, Current loss: {loss}")

                        if i % PARAMETERS.doValidation == 0:

                            correct, geomError, minErrorPossible = 0, 0, 0
                            descriptorsVM = []

                            for j, visualModelData in enumerate(visualModelDataloader, 0):
                                imgVM, coordsImgVM = visualModelData
                                imgVM = imgVM.to(device)

                                output = net(imgVM)
                                output = output.cpu()
                                descriptorsVM.append(output.detach().numpy()[0])

                            treeDesc = KDTree(descriptorsVM, leaf_size=2)

                            for j, validation_data in enumerate(validationDataloader, 0):

                                imgVal, coordsImgVal = validation_data
                                imgVal = imgVal.to(device)

                                output = net(imgVal)
                                output = output.cpu()
                                output = output.detach().numpy()[0]
                                coordsImgVal = coordsImgVal.detach().numpy()[0]

                                if lossFunction == 'circle loss' or lossFunction == 'angular loss':
                                    cosMax = 0
                                    for descVM in descriptorsVM:
                                        cosSimilarity = np.dot(output, descVM)
                                        if cosSimilarity > cosMax:
                                            cosMax = cosSimilarity
                                            idxMinPred = descriptorsVM.index(descVM)
                                else:
                                    _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                                    idxMinPred = idxDesc[0][0]

                                geomDistances, idxGeom = treeCoordsVM.query(coordsImgVal.reshape(1, -1), k=1)
                                idxMinReal = idxGeom[0][0]

                                coordsPredictedImg = coordsVM[idxMinPred]
                                coordsClosestImg = coordsVM[idxMinReal]

                                if idxMinPred in idxGeom[0]:
                                    correct += 1

                                geomError += np.linalg.norm(coordsImgVal - coordsPredictedImg)
                                minErrorPossible += np.linalg.norm(coordsImgVal - coordsClosestImg)

                            geomError /= len(validationDataloader)
                            minErrorPossible /= len(validationDataloader)
                            accuracy = correct * 100 / len(validationDataloader)

                            print(f"Average recall (k=1)= {accuracy}%")
                            print(f"Average geometric error: {geomError} m, Current error: {bestError} m")
                            print(f"Minimum reachable error: {minErrorPossible} m")
                            print(f"Relative error: {geomError - minErrorPossible} m\n")

                            if geomError <= bestError and geomError-minErrorPossible <= 0.10:
                                bestError = geomError
                                netName = os.path.join(netDir, "netLf_" + sl + "_" + room +
                                                       "_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netName)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {accuracy}%, Geometric error: {geomError} m\n\n")

                                writer.writerow([lossFunction, margin, room, epoch, i, accuracy, bestError])

                        if accuracy >= 100:
                            break

                    netName = os.path.join(netDir, "netLf_" + sl + "_" + room + "_ep_" + str(epoch) + "_end")
                    torch.save(net, netName)

                    if accuracy >= 100:
                        print(f"Training finished\n\n\n")
                        break
