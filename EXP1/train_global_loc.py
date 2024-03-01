"""
TRAIN CODE: GLOBAL LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the global method

Model:  vgg16
Pretrined? -> YES
Architecture: Triplet Network
One network is trained for all the rooms

Dataset: COLD-Freiburg, Seq2Cloudy3 Sampled (1 out of 5)
556 cloudy images
Training samples: random choice
Anchor, positive, negative -> same or different rooms
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos <= rNeg


Validation dataset: 556 cloudy images Seq2Cloudy3 Sampled (1 out of 5)
* The images are different from the training set
-each test image is compared with the images of the visual model of the entire map
-the nearest neighbour indicates the retrieved coordinates

Visual model dataset: the training set is employed as visual model
"""


import csv
import os
import numpy as np
from sklearn.neighbors import KDTree
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import losses
import create_datasets
import triplet_network
from config import PARAMETERS


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")

vmDataset = create_datasets.VisualModelGlobalLoc(imageFolderDataset=datasetDir + "/Train/")
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

coordsVM = []
for i, vmData in enumerate(vmDataloader, 0):
    _, coords = vmData
    coordsVM.append(coords.detach().numpy()[0])
treeCoordsVM = KDTree(coordsVM, leaf_size=2)

trainDataset = create_datasets.TrainGlobalLoc(
    tree=treeCoordsVM, imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"))
trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=16)

valDataset = create_datasets.ValidationGlobalLoc(imageFolderDataset=datasetDir + "/Validation/")
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp1TrainingDataGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Loss", "Margin", "Epoch", "Iteration", "Recall@k1", "Geometric Error"])

    selectedLosses = PARAMETERS.lossesGlobalLocTraining

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = PARAMETERS.losses.index(lossFunction)
        sl = PARAMETERS.lossAbreviations[idxLoss]
        margins = PARAMETERS.marginsGlobalLoc[idxLoss]
        lossDir = create_path(os.path.join(baseDir, "SAVED_MODELS", "EXP1", "GlobalLoc", sl))

        for margin in margins:
            marginDir = create_path(os.path.join(lossDir, str(margin)))

            net = triplet_network.TripletNetwork().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            """NETWORK TRAINING"""

            print("\nNEW TRAINING: ")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}\n")

            bestError = 1000
            epochs_since_improvement, early_stopping_threshold = 0, 3

            for epoch in range(PARAMETERS.numEpochsGlobalLoc):
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

                    """VALIDATION"""

                    if i % PARAMETERS.doValidation == 0:

                        recall, geomError, minErrorPossible = 0, 0, 0

                        descriptorsVM = []
                        for j, vmData in enumerate(vmDataloader, 0):
                            imgVM = vmData[0].to(device)
                            output = net(imgVM).cpu().detach().numpy()[0]
                            descriptorsVM.append(output)
                        treeDesc = KDTree(descriptorsVM, leaf_size=2)

                        for j, valData in enumerate(valDataloader, 0):

                            imgVal, coordsImgVal = valData
                            imgVal = imgVal.to(device)

                            output = net(imgVal).cpu().detach().numpy()[0]
                            coordsImgVal = coordsImgVal.detach().numpy()[0]

                            if lossFunction == 'circle loss' or lossFunction == 'angular loss':
                                cosSimilarities = np.dot(descriptorsVM, output)
                                predictedRoom = np.argmax(cosSimilarities)
                            else:
                                _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                                idxMinPred = idxDesc[0][0]

                            geomDistances, idxGeom = treeCoordsVM.query(coordsImgVal.reshape(1, -1), k=1)
                            idxMinReal = idxGeom[0][0]

                            coordsPredictedImg = coordsVM[idxMinPred]
                            coordsClosestImg = coordsVM[idxMinReal]

                            if idxMinPred in idxGeom[0]:
                                recall += 1

                            geomError += np.linalg.norm(coordsImgVal - coordsPredictedImg)
                            minErrorPossible += np.linalg.norm(coordsImgVal - coordsClosestImg)

                        recall *= 100 / len(valDataloader)
                        geomError /= len(valDataloader)
                        minErrorPossible /= len(valDataloader)

                        print(f"Average recall (k=1)= {recall}%")
                        print(f"Average geometric error: {geomError} m, Current error: {bestError} m")
                        print(f"Minimum reachable error: {minErrorPossible} m")
                        print(f"Relative error: {geomError - minErrorPossible}\n")

                        if geomError <= bestError:
                            bestError = geomError
                            epochs_since_improvement = 0

                            if geomError-minErrorPossible <= 0.10:

                                netDir = os.path.join(marginDir, "netLG_" + sl + "m" + str(margin) +
                                                      "_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netDir)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {recall}%, Geometric error: {geomError} m\n")

                                writer.writerow([lossFunction, margin, epoch, i + 1, recall, geomError])

                    if recall >= 100:
                        print("Training finished")
                        break
                netDir = os.path.join(marginDir, "netLG_" + sl + "m" + str(margin) + "_ep" + str(epoch) + "_end")
                torch.save(net, netDir)
                epochs_since_improvement += 1

                if epochs_since_improvement == early_stopping_threshold or recall >= 100:
                    print(f"Training finished\n\n\n")
                    break
