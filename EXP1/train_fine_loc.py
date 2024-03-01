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
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos <= rNeg


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
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import create_datasets
import losses
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


with open(csvDir + "/Exp1TrainingDataFineLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Loss", "Margin", "Room", "Epoch", "It", "Validation accuracy", "Geometric error"])

    selectedLosses = PARAMETERS.lossesFineLocTraining

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = PARAMETERS.losses.index(lossFunction)
        sl = PARAMETERS.lossAbreviations[idxLoss]
        margin = PARAMETERS.marginsFineLoc[idxLoss]

        lossDir_Lg = create_path(os.path.join(baseDir, "SAVED_MODELS", "EXP1", "CoarseLoc", "BestNets", sl))
        lossDir_Lf = create_path(os.path.join(baseDir, "SAVED_MODELS", "EXP1", "FineLoc", sl))
        trainNet = os.path.join(lossDir_Lg, os.listdir(lossDir_Lg)[0])

        trainDir = os.path.join(datasetDir, "Train")
        rooms = dset.ImageFolder(root=trainDir).classes

        for room in rooms:
            roomDir = create_path(os.path.join(lossDir_Lf, room))

            vmDataset = create_datasets.VisualModelTrainFineLoc(
                currentRoom=room, imageFolderDataset=datasetDir + "/Train/" + room)
            vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

            coordsVM = []
            for i, visualModelData in enumerate(vmDataloader, 0):
                _, coords = visualModelData
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

            trainDataset = create_datasets.TrainFineLoc(
                tree=treeCoordsVM, currentRoom=room, imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"))
            trainDataloader = DataLoader(trainDataset, shuffle=True, num_workers=0, batch_size=16)

            valDataset = create_datasets.ValidationFineLoc(
                currentRoom=room, imageFolderDataset=datasetDir + "/Validation/" + room)
            valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)

            net = torch.load(trainNet).to(device)

            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            """NETWORK TRAINING"""

            print("\nNew training: ")
            print(f"Network model: {trainNet}")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}")
            print(f"Room: {room}\n\n")

            bestError = 1000
            epochs_since_improvement, early_stopping_threshold = 0, 3

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

                        geomError /= len(valDataloader)
                        minErrorPossible /= len(valDataloader)
                        recall *= 100 / len(valDataloader)

                        print(f"Average recall (k=1)= {recall}%")
                        print(f"Average geometric error: {geomError} m, Current error: {bestError} m")
                        print(f"Minimum reachable error: {minErrorPossible} m")
                        print(f"Relative error: {geomError - minErrorPossible} m\n")

                        if geomError <= bestError:
                            bestError = geomError
                            epochs_since_improvement = 0

                            if geomError <= 0.30:
                                netDir = os.path.join(roomDir, "netLf_" + sl + "_" + room +
                                                      "_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netDir)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {recall}%, Geometric error: {geomError} m\n\n")

                                writer.writerow([lossFunction, margin, room, epoch, i, recall, bestError])

                    if recall >= 100:
                        break

                netDir = os.path.join(roomDir, "netLf_" + sl + "_" + room + "_ep_" + str(epoch) + "_end")
                torch.save(net, netDir)
                epochs_since_improvement += 1

                if epochs_since_improvement == early_stopping_threshold or recall >= 100:
                    print(f"Training finished\n\n\n")
                    break
