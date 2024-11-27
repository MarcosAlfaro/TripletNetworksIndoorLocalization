"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to train triplet networks to perform the coarse localization (first stage of hierarchical loc.)

Model: VGG16
Pretrained? -> YES (only the conv. layers)
Architecture: Triplet Network

Dataset: COLD-Freiburg, Seq2Cloudy3 Sampled (1 out of 5)
556 cloudy images
Training samples: random choice
Anchor, positive -> same room
Anchor, negative -> different room

Validation dataset: 556 cloudy images Seq2Cloudy3 Sampled (1 out of 5)
* The images are different from the training set
-each test image is compared with the representative image of every room
-the closest representative descriptor indicates the retrieved room

YAML PARAMETERS TO TAKE INTO ACCOUNT:
GPU device: device*
Directories: datasetDir*, csvDir*, modelsDir*
Batch size: batchSize
Training length: numEpochs, numIterations
Losses (and margins): selectedLosses, marginsCoarseLoc
*keep the same for all the scripts
"""


import torch
from torch.utils.data import DataLoader
import os
import csv
import numpy as np
from sklearn.neighbors import KDTree
import losses
import exp1_create_datasets
from models import VGG16
from config import PARAMS
from functions import create_path


# if the computer has cuda available, we will use cuda, else, cpu will be used
device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# parameter definition
csvDir = os.path.join(PARAMS.csvDir, "TRAIN_DATA")
N = PARAMS.batchSize
numEpochs, numIterations = PARAMS.numEpochs, PARAMS.numIterations
selectedLosses = PARAMS.selectedLosses

# generate the DataLoaders
trainDataset = exp1_create_datasets.TrainCoarseLoc()
valDataset = exp1_create_datasets.Validation()
imgRepDataset = exp1_create_datasets.RepImages()

trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=N)
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)


with open(csvDir + "/Exp1CoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Accuracy"])

    # a triplet network is trained for every loss function and margin value
    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = PARAMS.losses.index(lossFunction)
        sl = PARAMS.lossAbreviations[idxLoss]
        margins = PARAMS.marginsCoarseLoc[idxLoss]

        savedModelsDir = create_path(os.path.join(PARAMS.modelsDir, "EXP1", "HierarchicalLoc", "CoarseLoc"))
        lossDir = create_path(os.path.join(savedModelsDir, sl))

        for margin in margins:

            marginDir = create_path(os.path.join(lossDir, str(margin)))

            print("NEW TRAINING")
            print(f"Loss: {lossFunction}, margin: {margin}\n\n")

            # VGG16 model is employed as backbone
            net = VGG16().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            it, maxAccuracy = 0, 0
            for ep in range(numEpochs):

                for i, data in enumerate(trainDataloader, 0):
                    anc, pos, neg = data
                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

                    optimizer.zero_grad()

                    # the model receives three input images and outputs three descriptors
                    output1, output2, output3 = net(anc), net(pos), net(neg)

                    # the loss receives N*3 descriptors and outputs the loss value (float positive number)
                    loss = criterion(output1, output2, output3, margin)
                    loss.backward()

                    optimizer.step()

                    if i % (round(len(trainDataloader)/numIterations)) == 0:
                        print(f"Epoch {ep+1}/{numEpochs}, It {it%numIterations}/{numIterations}, Current loss: {loss}")

                        """VALIDATION"""

                        net.eval()
                        accuracy = 0

                        with torch.no_grad():

                            """REPRESENTATIVE DESCRIPTORS"""

                            descImgRep = []
                            for j, imgRepData in enumerate(imgRepDataloader, 0):
                                imgRep = imgRepData[0].to(device)
                                output = net(imgRep).cpu().detach().numpy()[0]
                                descImgRep.append(output)
                            treeImgRep = KDTree(descImgRep, leaf_size=2)

                            for j, valData in enumerate(valDataloader, 0):
                                imgVal, actualRoom, _ = valData
                                imgVal = imgVal.to(device)
                                output = net(imgVal).cpu().detach().numpy()[0]

                                # descriptors are compared via cosine similarity for these two losses
                                # otherwise, the Euclidean distance is used
                                if lossFunction in ['circle loss', 'angular loss']:
                                    cosSimilarities = np.dot(descImgRep, output)
                                    predictedRoom = np.argmax(cosSimilarities)
                                else:
                                    _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                                    predictedRoom = predictedRoom[0][0]

                                if predictedRoom == actualRoom[0]:
                                    accuracy += 1

                            accuracy *= 100 / len(valDataloader)

                            if accuracy > maxAccuracy:
                                maxAccuracy = accuracy

                            if i > 0 and accuracy > 98:
                                netDir = os.path.join(marginDir, "net_it" + str(it))
                                torch.save(net, netDir)

                            print(f"Validation accuracy: {accuracy} %, Max accuracy: {maxAccuracy} %\n")

                            writer.writerow([lossFunction, margin, it, accuracy])

                        net.train(True)
                        it += 1

                print(f"Epoch {ep + 1}/{numEpochs} finished")
                print(f"Validation accuracy: {accuracy} %, Max accuracy: {maxAccuracy} %\n")
                netDir = os.path.join(marginDir, "net_it" + str(it))
                torch.save(net, netDir)
