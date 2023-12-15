"""
TRAIN CODE: FINE LOC

AIM: analyze the influence of the triplet loss function on the performance of the network in the global method

Model:  vgg16
Pretrined? -> YES
Architecture: Triplet Network
One network is trained for all the rooms

Dataset: COLD-Freiburg, Seq2Cloudy3 Sampled (1 out of 5)
556 cloudy images
Training samples: random choice
Anchor, positive, negative -> same or different rooms
Anchor, positive -> d < rPos,  Anchor, negative -> d > rNeg,  rPos < rNeg


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
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader

import losses
import create_datasets
import triplet_network
from config import PARAMETERS


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    run()


visualModelDataset = create_datasets.VisualModelGlobalLoc(imageFolderDataset=datasetDir + "/Entrenamiento/",
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

trainingDataset = create_datasets.TrainGlobalLoc(imageFolderDataset=dset.ImageFolder(datasetDir + "/Entrenamiento/"),
                                                 transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                               transforms.ToTensor()
                                                                               ]),
                                                 should_invert=False)

trainDataloader = DataLoader(trainingDataset, shuffle=False, num_workers=0, batch_size=16)


valDataset = create_datasets.ValidationGlobalLoc(imageFolderDataset=datasetDir + "/Validacion/",
                                                 transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                               transforms.ToTensor()
                                                                               ]),
                                                 should_invert=False)

valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


# print('Training batch number: {}'.format(len(train_dataloader)))

"""ENTRENAMIENTO DE LA RED"""

with open(csvDir + "/TrainingDataGlobalLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Loss", "Margin", "Epoch", "Iteration", "Recallk1", "Geometric Error"])

    selectedLosses = PARAMETERS.lossesGlobalLocTraining

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
        margins = PARAMETERS.marginsGlobalLoc[idxLoss]

        for margin in margins:

            if not os.path.exists(os.path.join(baseDir, "SAVED_MODELS", "GlobalLoc", sl, str(margin))):
                os.mkdir(os.path.join(baseDir, "SAVED_MODELS", "GlobalLoc", sl, str(margin)))
            netDir = os.path.join(baseDir, "SAVED_MODELS", "GlobalLoc", sl, str(margin))

            bestError = 1000
            i, j = 0, 0

            net = triplet_network.TripletNetwork().to(device)
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            print("\nNEW TRAINING: ")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}\n")

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

                        correct, geomError, minErrorPossible = 0, 0, 0
                        descriptorsVM = []

                        for j, vmData in enumerate(visualModelDataloader, 0):

                            imgVM, coordsImgVM = vmData
                            imgVM = imgVM.to(device)

                            output = net(imgVM)
                            output = output.cpu()
                            descriptorsVM.append(output.detach().numpy()[0])

                        treeDesc = KDTree(descriptorsVM, leaf_size=2)

                        for j, valData in enumerate(valDataloader, 0):

                            imgVal, coordsImgVal = valData
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

                        accuracy = (correct / (j+1)) * 100
                        geomError /= (j+1)
                        minErrorPossible /= (j+1)

                        print(f"Average recall (k=1)= {accuracy}%")
                        print(f"Average geometric error: {geomError} m, Current error: {bestError} m")
                        print(f"Minimum reachable error: {minErrorPossible} m")
                        print(f"Relative error: {geomError - minErrorPossible}\n")

                        if geomError <= bestError:
                            bestError = geomError

                            if geomError-minErrorPossible <= 0.10:

                                netName = os.path.join(netDir, "netLG_" + sl + "m" + str(margin) +
                                                       "_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netName)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {accuracy}%, Geometric error: {geomError} m\n")

                                writer.writerow([lossFunction, margin, epoch, i + 1, accuracy, geomError])

                    if accuracy >= 100:
                        print("Training finished")
                        break
                netName = os.path.join(netDir, "netLG_" + sl + "m" + str(margin) + "_ep" + str(epoch) + "_end")
                torch.save(net, netName)

                if accuracy <= 100:
                    break
