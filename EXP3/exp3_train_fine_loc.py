"""
EXPERIMENT 3:
-evaluation of triplet networks in different environments simultaneously

This script is used to train triplet networks to perform the fine localization (second stage of hierarchical loc.)

Model: VGG16
Pretrained? -> YES (only the conv. layers)
Architecture: Triplet Netwo

Train dataset:
COLD-Freiburg, Seq2Cloudy3 Sampled (588 images)
COLD-Saarbrücken (Part A) Seq2Cloudy3 Sampled (586 images)
COLD-Saarbrücken (Part B) Seq4Cloudy1 Sampled (321 images)
TOTAL: 1495 cloudy images
Training samples: random choice
Anchor, positive, negative -> same room
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos < rNeg

Validation dataset:
COLD-Freiburg, Seq2Cloudy3 Sampled (586 images)
COLD-Saarbrücken (Part A) Seq2Cloudy3 Sampled (582 images)
COLD-Saarbrücken (Part B) Seq4Cloudy1 Sampled (301 images)
TOTAL: 1469 cloudy images
* The images are different from the training set
-each test image is compared with the images of the visual model of the room
-the nearest neighbor indicates the retrieved coordinates

Visual model dataset: the training set is employed as visual model

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: datasetDir*, csvDir*, modelsDir*
Batch size: batchSize
Training length: numEpochs, numIterations
Losses (and margins): selectedLosses, marginsCoarseLoc**
* keep the same for all the scripts
** use the same margin values as in coarse loc.
"""

import csv
import os
import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import exp3_create_datasets
import losses
from config import PARAMS
from functions import create_path


device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')

csvDir = os.path.join(PARAMS.csvDir, "TRAIN_DATA")
datasetDir = os.path.join(PARAMS.datasetDir, "3ENVIRONMENTS")
numEpochs, numIterations = PARAMS.numEpochs, PARAMS.numIterations

selectedLosses = PARAMS.selectedLosses
N = PARAMS.batchSize


with open(csvDir + "/FineLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Room", "Epoch", "It", "Validation accuracy", "Geometric error"])

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = PARAMS.losses.index(lossFunction)
        sl = PARAMS.lossAbreviations[idxLoss]
        margin = PARAMS.marginsFineLoc[idxLoss]

        savedModelsDir = create_path(os.path.join(PARAMS.modelsDir, "EXP3", "HierarchicalLoc", "FineLoc"))
        lossDir_FL = create_path(os.path.join(savedModelsDir, sl))

        lossDir_CL = os.path.join(PARAMS.modelsDir, "EXP3", "HierarchicalLoc", "CoarseLoc", sl, str(margin))
        trainNet = os.path.join(lossDir_CL, os.listdir(lossDir_CL)[0])

        trainDir = os.path.join(datasetDir, "Train")
        rooms = dset.ImageFolder(root=trainDir).classes

        for room in rooms:
            roomDir = create_path(os.path.join(lossDir_FL, room))

            vmDataset = exp3_create_datasets.VisualModel()
            vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

            coordsVM = []
            for i, vmData in enumerate(vmDataloader, 0):
                _, actualRoom, coords = vmData
                if actualRoom[0] != room:
                    continue
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

            trainDataset = exp3_create_datasets.TrainFineLoc(currentRoom=room)
            trainDataloader = DataLoader(trainDataset, shuffle=True, num_workers=0, batch_size=16)

            valDataset = exp3_create_datasets.Validation()
            valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)

            net = torch.load(trainNet).to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            """NETWORK TRAINING"""

            print("NEW TRAINING")
            print(f"Loss: {lossFunction}, margin: {margin}, Room: {room}\n\n")

            it, bestError = 0, 1000
            for ep in range(numEpochs):

                for i, data in enumerate(trainDataloader, 0):
                    anc, pos, neg = data
                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

                    optimizer.zero_grad()

                    output1, output2, output3 = net(anc), net(pos), net(neg)
                    loss = criterion(output1, output2, output3, margin)
                    loss.backward()

                    optimizer.step()

                    if i % (round(len(trainDataloader)/numIterations)) == 0:
                        print(f"Epoch {ep+1}/{numEpochs}, It {it%numIterations}/{numIterations}, Current loss: {loss}")

                        """VALIDATION"""

                        net.eval()
                        with torch.no_grad():
                            recall, geomError, minErrorPossible = 0, 0, 0

                            descriptorsVM = []
                            for j, vmData in enumerate(vmDataloader, 0):
                                imgVM, actualRoom, _ = vmData
                                if actualRoom[0] == room:
                                    imgVM = imgVM.to(device)
                                    output = net(imgVM).cpu().detach().numpy()[0]
                                    descriptorsVM.append(output)
                            treeDesc = KDTree(descriptorsVM, leaf_size=2)

                            numImgsRoom = 0
                            for j, valData in enumerate(valDataloader, 0):
                                imgVal, actualRoom, coordsImgVal = valData
                                if actualRoom[0] == room:
                                    numImgsRoom += 1
                                    imgVal = imgVal.to(device)
                                    output = net(imgVal).cpu().detach().numpy()[0]
                                    coordsImgVal = coordsImgVal.detach().numpy()[0]

                                    if lossFunction in ['circle loss', 'angular loss']:
                                        cosSimilarities = np.dot(descriptorsVM, output)
                                        idxMinPred = np.argmax(cosSimilarities)
                                    else:
                                        _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                                        idxMinPred = idxDesc[0][0]

                                    geomDistances, idxGeom = treeCoordsVM.query(coordsImgVal.reshape(1, -1), k=1)
                                    idxMinReal = idxGeom[0][0]

                                    coordsPredictedImg, coordsClosestImg = coordsVM[idxMinPred], coordsVM[idxMinReal]

                                    if idxMinPred in idxGeom[0]:
                                        recall += 1

                                    geomError += np.linalg.norm(coordsImgVal - coordsPredictedImg)
                                    minErrorPossible += np.linalg.norm(coordsImgVal - coordsClosestImg)

                            recall *= 100 / numImgsRoom
                            geomError /= numImgsRoom
                            minErrorPossible /= numImgsRoom

                            if geomError <= bestError:
                                bestError = geomError

                            if i > 0:
                                netDir = os.path.join(roomDir, "net_it" + str(it))
                                torch.save(net, netDir)

                            print(f"Recall@1: {recall} %, Geometric error: {geomError} m, Best error: {bestError} m\n")

                            writer.writerow([lossFunction, margin, it, recall, geomError])
                            net.train(True)
                            it += 1

                print(f"Epoch {ep + 1}/{numEpochs} finished")
                netDir = os.path.join(roomDir, "net_it" + str(it))
                torch.save(net, netDir)
