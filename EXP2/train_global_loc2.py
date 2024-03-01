"""
TRAIN CODE: GLOBAL LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the global method

In this experiment, three different environments are considered: Friburgo, Saarbrücken A & Saarbrücken B

Model:  vgg16
Pretrined? -> YES
Architecture: Triplet Network
One network is trained for all the rooms

Train dataset:
COLD-Freiburg, Seq2Cloudy3 Sampled (588 images)
COLD-Saarbrücken (Part A) Seq2Cloudy3 Sampled (586 images)
COLD-Saarbrücken (Part B) Seq4Cloudy1 Sampled (321 images)
TOTAL: 1495 cloudy images
Training samples: random choice
Anchor, positive, negative -> same or different rooms
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos < rNeg


Validation dataset: 509 cloudy images
COLD-Freiburg, Seq2Cloudy3 Sampled (199 images)
COLD-Saarbrücken (Part A) Seq2Cloudy3 Sampled (198 images)
COLD-Saarbrücken (Part B) Seq4Cloudy1 Sampled (112 images)
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
import create_datasets2
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
datasetDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS")


vmDataset = create_datasets2.VisualModelGlobalLoc(imageFolderDataset=datasetDir + "/Train/")
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

treeCoordsVMenv = []
coordsVMenv = [[], [], []]
coordsVM = []
for i, vmData in enumerate(vmDataloader, 0):
    _, idxEnv, coords = vmData
    idxEnv = idxEnv.detach().numpy()[0]
    coordsVM.append(coords.detach().numpy()[0])
    coordsVMenv[idxEnv].append(coords.detach().numpy()[0])
for env in range(3):
    treeCoordsVMenv.append(KDTree(coordsVMenv[env], leaf_size=2))
treeCoordsVM = KDTree(coordsVM, leaf_size=2)

trainDataset = create_datasets2.TrainGlobalLoc(
    tree=treeCoordsVM, imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"))
trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=16)

valDataset = create_datasets2.ValidationGlobalLoc(imageFolderDataset=datasetDir + "/Validation/")
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp2TrainingDataGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Loss", "Margin", "Epoch", "Iteration", "Recallk1", "Geometric Error"])

    selectedLosses = PARAMETERS.lossesGlobalLocTraining

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = PARAMETERS.losses.index(lossFunction)
        sl = PARAMETERS.lossAbreviations[idxLoss]
        margins = PARAMETERS.marginsGlobalLoc[idxLoss]
        lossDir = create_path(os.path.join(baseDir, "SAVED_MODELS", "EXP2", "GlobalLoc", sl))

        for margin in margins:
            marginDir = create_path(os.path.join(lossDir, str(margin)))

            net = triplet_network.TripletNetwork().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            print("\nNEW TRAINING: ")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}\n")

            bestError = 1000
            epochs_since_improvement = 0
            early_stopping_threshold = 3

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
                    recall, accuracyEnv = 0, 0
                    if i % PARAMETERS.doValidation == 0:

                        correct, geomError, minErrorPossible = 0, 0, 0
                        descriptorsVM, idxsEnv = [], []

                        for j, vmData in enumerate(vmDataloader, 0):
                            imgVM, idxEnvVM, coordsImgVM = vmData
                            imgVM = imgVM.to(device)

                            output = net(imgVM).cpu().detach().numpy()[0]
                            descriptorsVM.append(output)
                            idxsEnv.append(idxEnvVM.detach().numpy()[0])

                        treeDesc = KDTree(descriptorsVM, leaf_size=2)

                        for j, valData in enumerate(valDataloader, 0):
                            imgVal, actualEnv, coordsImgVal = valData
                            imgVal = imgVal.to(device)

                            output = net(imgVal).cpu().detach().numpy()[0]
                            coordsImgVal = coordsImgVal.detach().numpy()[0]
                            actualEnv = actualEnv.detach().numpy()[0]

                            if lossFunction == 'circle loss' or lossFunction == 'angular loss':
                                cosSimilarities = np.dot(descriptorsVM, output)
                                idxMinPred = np.argmax(cosSimilarities)
                            else:
                                _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                                idxMinPred = idxDesc[0][0]

                            geomDistances, idxGeom = treeCoordsVMenv[actualEnv].query(coordsImgVal.reshape(1, -1), k=1)
                            idxMinReal = idxGeom[0][0]

                            coordsPredictedImg = coordsVM[idxMinPred]
                            coordsClosestImg = coordsVMenv[actualEnv][idxMinReal]
                            predEnv = idxsEnv[idxMinPred]
                            if predEnv == actualEnv:
                                accuracyEnv += 1
                                if idxMinPred in idxGeom[0]:
                                    recall += 1

                            geomError += np.linalg.norm(coordsImgVal - coordsPredictedImg)
                            minErrorPossible += np.linalg.norm(coordsImgVal - coordsClosestImg)

                        geomError /= accuracyEnv
                        recall *= 100 / len(valDataloader)
                        accuracyEnv *= 100 / len(valDataloader)

                        minErrorPossible /= len(valDataloader)

                        print(f"Average recall (k=1)= {recall} %")
                        print(f"Average geometric error: {geomError} m, Current error: {bestError} m")
                        print(f"Minimum reachable error: {minErrorPossible} m")
                        print(f"Relative error: {geomError - minErrorPossible}\n")

                        if geomError <= bestError:
                            bestError = geomError
                            epochs_since_improvement = 0

                            if geomError <= 0.30:
                                netDir = os.path.join(marginDir, "netLG_" + sl + "m" + str(margin) +
                                                      "_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netDir)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {recall} %, Geometric error: {geomError} m\n")

                                writer.writerow([lossFunction, margin, epoch, i + 1, recall, geomError])

                    if recall >= 100:
                        break
                netDir = os.path.join(marginDir, "netLG_" + sl + "m" + str(margin) + "_ep" + str(epoch) + "_end")
                torch.save(net, netDir)
                epochs_since_improvement += 1

                if recall >= 100 or epochs_since_improvement == early_stopping_threshold:
                    print("Training finished")
                    break
