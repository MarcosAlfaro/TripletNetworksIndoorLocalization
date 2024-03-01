"""
TRAIN CODE: COARSE LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the room retrieval task

Model: vgg16
Pretrained? -> YES
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

"""

import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import os
import csv
import numpy as np
from sklearn.neighbors import KDTree
import losses
import create_datasets
import triplet_network
from config import PARAMETERS


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


# if the computer has cuda available, we will use cuda, else, cpu will be used
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")


trainDataset = create_datasets.TrainCoarseLoc(imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"))
valDataset = create_datasets.ValidationCoarseLoc(imageFolderDataset=datasetDir + "/Validation/")
imgRepDataset = create_datasets.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/")

# we load the image sets into the gpu/cpu
trainDataloader = DataLoader(trainDataset, shuffle=True, num_workers=0, batch_size=16)
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp1TrainingDataCoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Accuracy"])

    selectedLosses = PARAMETERS.lossesCoarseLocTraining

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = PARAMETERS.losses.index(lossFunction)
        sl = PARAMETERS.lossAbreviations[idxLoss]
        margins = PARAMETERS.marginsCoarseLoc[idxLoss]
        lossDir = create_path(os.path.join(baseDir, "SAVED_MODELS", "EXP1", "CoarseLoc", sl))

        for margin in margins:

            print("NEW TRAINING")
            print(f"Loss: {lossFunction}, margin: {margin}\n\n")

            net = triplet_network.TripletNetwork().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            marginDir = create_path(os.path.join(lossDir, str(margin)))

            maxAccuracy = 0
            epochs_since_improvement, early_stopping_threshold = 0, 3

            for epoch in range(0, PARAMETERS.numEpochsCoarseLoc):

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
                        accuracy = 0

                        """REPRESENTATIVE DESCRIPTORS"""

                        descImgRep = []
                        for j, imgRepData in enumerate(imgRepDataloader, 0):
                            imgRep = imgRepData[0].to(device)
                            output = net(imgRep).cpu().detach().numpy()[0]
                            descImgRep.append(output)
                        treeImgRep = KDTree(descImgRep, leaf_size=2)

                        for j, valData in enumerate(valDataloader, 0):
                            imgVal, actualRoom = valData
                            imgVal = imgVal.to(device)
                            output = net(imgVal).cpu().detach().numpy()[0]

                            if lossFunction == 'circle loss' or lossFunction == 'angular loss':
                                cosSimilarities = np.dot(descImgRep, output)
                                predictedRoom = np.argmax(cosSimilarities)
                            else:
                                _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                                predictedRoom = predictedRoom[0][0]

                            if predictedRoom == actualRoom[0]:
                                accuracy += 1

                        accuracy *= 100 / len(valDataloader)
                        print(f"Validation Accuracy = {accuracy} %\n")

                        if accuracy > maxAccuracy:
                            maxAccuracy = accuracy
                            epochs_since_improvement = 0

                            if accuracy > 90:
                                netDir = os.path.join(marginDir, "netLg_" + sl + "m" + str(margin) +
                                                      "ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netDir)

                                print("NETWORK SAVED")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation accuracy: {maxAccuracy}%\n")

                                writer.writerow([lossFunction, margin, epoch, i, maxAccuracy])

                        if accuracy >= 100:
                            break

                netDir = os.path.join(marginDir, "netLg_" + sl + "m" + str(margin) + "ep" + str(epoch) + "end")
                torch.save(net, netDir)
                epochs_since_improvement += 1

                if epochs_since_improvement == early_stopping_threshold or accuracy >= 100:
                    print("Training finished")
                    break
