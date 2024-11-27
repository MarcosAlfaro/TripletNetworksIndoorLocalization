"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to train triplet networks to perform the global localization

Model: VGG16
Pretrained? -> YES (only the conv. layers)
Architecture: Triplet Network

Dataset: COLD-Freiburg, Seq2Cloudy3 Sampled (1 out of 5)
556 cloudy images
Training samples: random choice
Anchor, positive, negative -> same or different rooms
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos <= rNeg

Validation dataset: 556 cloudy images Seq2Cloudy3 Sampled (1 out of 5)
* The images are different from the training set
-each test image is compared with the images of the visual model of the entire map
-the nearest neighbor indicates the retrieved coordinates

Visual model dataset: the training set is employed as visual model

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: datasetDir*, csvDir*, modelsDir*
Batch size: batchSize
Training length: numEpochs, numIterations
Losses (and margins): selectedLosses, marginsCoarseLoc
* keep the same for all the scripts
"""


import csv
import os
import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import DataLoader
import losses
import exp1_create_datasets
from config import PARAMS
from models import VGG16
from functions import create_path

device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = os.path.join(PARAMS.csvDir, "TRAIN_DATA")
selectedLosses = PARAMS.selectedLosses
N = PARAMS.batchSize
numEpochs, numIterations = PARAMS.numEpochs, PARAMS.numIterations

vmDataset = exp1_create_datasets.VisualModel()
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

coordsVM = []
for i, vmData in enumerate(vmDataloader, 0):
    _, _, coords = vmData
    coordsVM.append(coords.detach().numpy()[0])
treeCoordsVM = KDTree(coordsVM, leaf_size=2)

trainDataset = exp1_create_datasets.TrainGlobalLoc()
trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=N)

valDataset = exp1_create_datasets.Validation()
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp1GlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Recall@k1", "Geometric Error"])

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        savedModelsDir = create_path(os.path.join(PARAMS.modelsDir, "EXP1", "GlobalLoc"))

        idxLoss = PARAMS.losses.index(lossFunction)
        sl = PARAMS.lossAbreviations[idxLoss]
        margins = PARAMS.marginsGlobalLoc[idxLoss]
        lossDir = create_path(os.path.join(savedModelsDir, sl))

        for margin in margins:

            marginDir = create_path(os.path.join(lossDir, str(margin)))

            net = VGG16().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            """NETWORK TRAINING"""

            print("\nNEW TRAINING: ")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}\n")


            it, bestError = 0, 1000
            for ep in range(PARAMS.numEpochs):

                for i, data in enumerate(trainDataloader, 0):
                    anc, pos, neg = data
                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

                    optimizer.zero_grad()

                    output1, output2, output3 = net(anc), net(pos), net(neg)
                    loss = criterion(output1, output2, output3, margin)
                    loss.backward()

                    optimizer.step()

                    if i % (round(len(trainDataloader)/numIterations)) == 0:
                        print(f"Epoch {ep+1}/{numEpochs}, It {it%numIterations}/{numIterations}, "
                              f"Current loss: {loss}")

                        recall, geomError, minErrorPossible = 0, 0, 0
                        net.eval()

                        with torch.no_grad():

                            """VISUAL MODEL"""
                            descVM = []
                            for j, vmData in enumerate(vmDataloader, 0):
                                imgVM = vmData[0].to(device)
                                output = net(imgVM).cpu().detach().numpy()[0]
                                descVM.append(output)
                            treeDesc = KDTree(descVM, leaf_size=2)

                            """VALIDATION"""
                            for j, valData in enumerate(valDataloader, 0):

                                imgVal, _, coordsImgVal = valData
                                imgVal = imgVal.to(device)

                                out_gpu = net(imgVal)
                                out_cpu = net(imgVal).cpu().detach().numpy()[0]
                                coordsImgVal = coordsImgVal.detach().numpy()[0]

                                if lossFunction in ['circle loss', 'angular loss']:
                                    cosSimilarities = np.dot(descVM, output)
                                    idxMinPred = np.argmax(cosSimilarities)
                                else:
                                    _, idxDesc = treeDesc.query(out_cpu.reshape(1, -1), k=1)
                                    idxMinPred = idxDesc[0][0]

                                geomDistances, idxGeom = treeCoordsVM.query(coordsImgVal.reshape(1, -1), k=1)
                                idxMinReal = idxGeom[0][0]

                                coordsPredictedImg, coordsClosestImg = coordsVM[idxMinPred], coordsVM[idxMinReal]

                                if idxMinPred in idxGeom[0]:
                                    recall += 1

                                geomError += np.linalg.norm(coordsImgVal - coordsPredictedImg)
                                minErrorPossible += np.linalg.norm(coordsImgVal - coordsClosestImg)

                            recall *= 100 / len(valDataloader)
                            geomError /= len(valDataloader)
                            minErrorPossible /= len(valDataloader)

                            if geomError <= bestError:
                                bestError = geomError

                            if i > 0 and geomError <= 0.15:
                                netDir = os.path.join(marginDir, "net_it" + str(it))
                                torch.save(net, netDir)

                            print(f"Recall@1: {recall} %, Geometric error: {geomError} m, Best error: {bestError} m\n")
                            writer.writerow([lossFunction, margin, it, recall, geomError])

                        net.train(True)
                        it += 1

                print(f"Epoch {ep + 1}/{numEpochs} finished")
                print(f"Recall@1: {recall} %, Geometric error: {geomError} m, Best error: {bestError} m\n")
                netDir = os.path.join(marginDir, "net_end")
                torch.save(net, netDir)
