"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to test siamese networks to perform the global localization
"""


import csv
import os
import numpy as np
from sklearn.neighbors import KDTree
import torch
from torch.utils.data import DataLoader
import losses
import exp1_create_datasets
from models import VGG16
from functions import create_path
from config import PARAMS


device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

csvDir = os.path.join(PARAMS.csvDir, "TRAIN_DATA")
N = PARAMS.batchSize
numEpochs, numIterations = PARAMS.numEpochs, PARAMS.numIterations

vmDataset = exp1_create_datasets.VisualModel()
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

coordsVM = []
for i, vmData in enumerate(vmDataloader, 0):
    _, _, coords = vmData
    coordsVM.append(coords.detach().numpy()[0])
treeCoordsVM = KDTree(coordsVM, leaf_size=2)

trainDataset = exp1_create_datasets.SNNTrainGlobalLoc()
trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=N)

valDataset = exp1_create_datasets.Validation()
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


"""NETWORK TRAINING"""

with open(csvDir + "/SNNGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Recall@k1", "Geometric Error"])

    criterion = losses.ContrastiveLoss()

    savedModelsDir = create_path(os.path.join(PARAMS.modelsDir, "EXP1", "GlobalLoc"))

    idxLoss = PARAMS.losses.index("contrastive loss")
    sl = PARAMS.lossAbreviations[idxLoss]
    margin = PARAMS.marginsGlobalLoc[idxLoss][0]
    lossDir = create_path(os.path.join(savedModelsDir, sl))

    marginDir = create_path(os.path.join(savedModelsDir, sl, str(margin)))

    net = VGG16().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    """NETWORK TRAINING"""

    print("\nNEW TRAINING: ")
    print(f"Loss: Contrastive Loss, margin/alpha: {margin} (Siamese Network)\n\n")

    it, bestError = 0, 1000
    for ep in range(numEpochs):

        for i, data in enumerate(trainDataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            optimizer.zero_grad()

            output1, output2 = net(img0), net(img1)
            loss = criterion(output1, output2, label)
            loss.backward()

            optimizer.step()

            if i % (round(len(trainDataloader) / numIterations)) == 0:
                print(f"Epoch {ep+1}/{numEpochs}, It {it % numIterations}/{numIterations}, Current loss: {loss}")

                """VALIDATION"""

                net.eval()
                with torch.no_grad():
                    recall, geomError, minErrorPossible = 0, 0, 0

                    descVM = []
                    for j, vmData in enumerate(vmDataloader, 0):
                        imgVM = vmData[0].to(device)
                        output = net(imgVM).cpu().detach().numpy()[0]
                        descVM.append(output)
                    treeDesc = KDTree(descVM, leaf_size=2)

                    for j, valData in enumerate(valDataloader, 0):

                        imgVal, _, coordsImgVal = valData
                        imgVal = imgVal.to(device)

                        output = net(imgVal).cpu().detach().numpy()[0]
                        coordsImgVal = coordsImgVal.detach().numpy()[0]

                        _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
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

                    if i > 0 and geomError <= 5:
                        netDir = os.path.join(marginDir, "net_it" + str(it))
                        torch.save(net, netDir)

                    print(f"Recall@1: {recall} %, Geometric error: {geomError} m, Best error: {bestError} m\n")

                    writer.writerow(["Contrastive Loss", margin, it, recall, geomError])

                net.train(True)
                it += 1

        print(f"Epoch {ep+1}/{numEpochs} finished")
        print(f"Recall@1: {recall} %, Geometric error: {geomError} m, Best error: {bestError} m\n")
        netDir = os.path.join(marginDir, "net_end")
        torch.save(net, netDir)
