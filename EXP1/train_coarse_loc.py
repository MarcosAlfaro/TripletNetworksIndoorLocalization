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
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
import csv
import numpy as np
from sklearn.neighbors import KDTree
import losses
import create_datasets
import triplet_network
from config import PARAMETERS

# if the computer has cuda available, we will use cuda, else, cpu will be used
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


trainingDataset = create_datasets.TrainCoarseLoc(imageFolderDataset=dset.ImageFolder(datasetDir + "/Entrenamiento/"),
                                                 transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                               transforms.ToTensor()
                                                                               ]),
                                                 should_invert=False)

validationDataset = create_datasets.ValidationCoarseLoc(imageFolderDataset=datasetDir + "/Validacion/",
                                                        transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                      transforms.ToTensor()
                                                                                      ]),
                                                        should_invert=False)


imgRepDataset = create_datasets.RepresentativeImages(imageFolderDataset=datasetDir + "/RepresentativeImages/",
                                                     transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                   transforms.ToTensor()
                                                                                   ]),
                                                     should_invert=False)

# we load the image sets into the gpu/cpu
trainDataloader = DataLoader(trainingDataset, shuffle=True, num_workers=0, batch_size=16)

valDataloader = DataLoader(validationDataset, shuffle=False, num_workers=0, batch_size=1)

imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

print('Training batch number: {}'.format(len(trainDataloader)))

# print(net)
# params = list(net.parameters())
# print(f'The number of net parameters is: {len(params)}')


"""NETWORK TRAINING"""

with open(csvDir + "/TrainingDataCoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Accuracy"])

    selectedLosses = PARAMETERS.lossesCoarseLocTraining

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
        margins = PARAMETERS.marginsCoarseLoc[idxLoss]

        for margin in margins:

            print("NEW TRAINING")
            print(f"Loss: {lossFunction}, margin: {margin}\n\n")

            net = triplet_network.TripletNetwork().to(device)
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            if not os.path.exists(os.path.join(baseDir, "SAVED_MODELS", "CoarseLoc", sl, str(margin))):
                os.mkdir(os.path.join(baseDir, "SAVED_MODELS", "CoarseLoc", sl, str(margin)))
            netDir = os.path.join(baseDir, "SAVED_MODELS", "CoarseLoc", sl, str(margin))

            i, j = 0, 0
            maxAccuracy = 0

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
                        correct = 0

                        """REPRESENTATIVE DESCRIPTORS"""

                        descImgRep = []

                        for j, imgRepData in enumerate(imgRepDataloader, 0):

                            imgRep, _ = imgRepData
                            imgRep = imgRep.to(device)

                            output = net(imgRep)
                            output = output.cpu()
                            descImgRep.append(output.detach().numpy()[0])

                        treeImgRep = KDTree(descImgRep, leaf_size=2)

                        for j, valData in enumerate(valDataloader, 0):

                            imgVal, actualRoom = valData
                            imgVal = imgVal.to(device)

                            output = net(imgVal)
                            output = output.cpu()
                            output = output.detach().numpy()[0]

                            if lossFunction == 'circle loss' or lossFunction == 'angular loss':

                                cosMax = 0
                                for desc in descImgRep:
                                    cosSimilarity = np.dot(desc, output)
                                    if cosSimilarity > cosMax:
                                        cosMax = cosSimilarity
                                        predictedRoom = descImgRep.index(desc)
                            else:
                                _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                                predictedRoom = predictedRoom[0][0]

                            if predictedRoom == actualRoom[0]:
                                correct += 1

                        accuracy = correct * 100 / len(valDataloader)

                        print(f"Validation Accuracy= {accuracy} %\n")

                        if accuracy >= maxAccuracy and accuracy > 90:

                            maxAccuracy = accuracy

                            netName = os.path.join(netDir, "netLg_" + sl + "m" + str(margin) +
                                                   "ep" + str(epoch) + "it" + str(i))

                            torch.save(net, netName)

                            print("NETWORK SAVED")
                            print(f"Epoch: {epoch}, It: {i}")
                            print(f"Validation accuracy: {maxAccuracy}%\n")

                            writer.writerow([lossFunction, margin, epoch, i, maxAccuracy])

                    if accuracy >= 100:
                        print(f"Training finished")
                        break
                    netName = os.path.join(netDir, "netLg_" + sl + "m" + str(margin) + "ep" + str(epoch) + "end")
                    torch.save(net, netName)

                if accuracy >= 100:
                    break
