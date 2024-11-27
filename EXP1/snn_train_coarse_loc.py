"""
EXPERIMENT 1:
-comparison between two localization approaches: hierarchical and global
-comparison among different triplet loss functions

This script is used to test siamese networks to perform the coarse localization (first stage of hierarchical loc.)

"""
import torch
from torch.utils.data import DataLoader
import os
import csv
from sklearn.neighbors import KDTree
import losses
import exp1_create_datasets
from models import VGG16
from config import PARAMS


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = os.path.join(PARAMS.csvDir, "TRAIN_DATA")
N = PARAMS.batchSize
numEpochs, numIterations = PARAMS.numEpochs, PARAMS.numIterations

trainDataset = exp1_create_datasets.SNNTrainCoarseLoc()
valDataset = exp1_create_datasets.Validation()
imgRepDataset = exp1_create_datasets.RepImages()


trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=N)
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)
imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)


"""NETWORK TRAINING"""

with open(csvDir + "/SNNCoarseLoc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Accuracy"])

    criterion = losses.ContrastiveLoss()
    idxLoss = PARAMS.losses.index("contrastive loss")
    sl = PARAMS.lossAbreviations[idxLoss]
    margin = PARAMS.marginsCoarseLoc[idxLoss][0]

    savedModelsDir = create_path(os.path.join(PARAMS.modelsDir, "EXP1", "HierarchicalLoc", "CoarseLoc", sl))
    marginDir = create_path(os.path.join(savedModelsDir, str(margin)))


    print("NEW TRAINING")
    print(f"Loss: Contrastive Loss , margin: {margin} (Siamese Network)\n\n")

    net = VGG16().to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    it, maxAccuracy = 0, 0
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
                print(f"Epoch {ep + 1}/{numEpochs}, It {it % numIterations}/{numIterations}, Current loss: {loss}")

                accuracy = 0
                net.eval()
                with torch.no_grad():

                    """REPRESENTATIVE DESCRIPTORS"""

                    descImgRep = []
                    for j, imgRepData in enumerate(imgRepDataloader, 0):
                        imgRep = imgRepData[0].to(device)
                        output = net(imgRep).cpu().detach().numpy()[0]
                        descImgRep.append(output)
                    treeImgRep = KDTree(descImgRep, leaf_size=2)

                    """VALIDATION"""

                    for j, valData in enumerate(valDataloader, 0):
                        imgVal, actualRoom, _ = valData
                        imgVal = imgVal.to(device)
                        output = net(imgVal).cpu().detach().numpy()[0]

                        _, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=1)
                        predictedRoom = predictedRoom[0][0]

                        if predictedRoom == actualRoom[0]:
                            accuracy += 1

                    accuracy *= 100 / len(valDataloader)

                    if accuracy > maxAccuracy:
                        maxAccuracy = accuracy

                    if i > 0 and accuracy > 20:
                        netDir = os.path.join(marginDir, "net_it" + str(it))
                        torch.save(net, netDir)

                    print(f"Validation accuracy: {accuracy} %, Max accuracy: {maxAccuracy} %\n")
                    writer.writerow(["Contrastive Loss", margin, it, accuracy])

                net.train(True)
                it += 1

        print(f"Epoch {ep + 1}/{numEpochs} finished")
        print(f"Validation accuracy: {accuracy} %, Max accuracy: {maxAccuracy} %\n")
        netDir = os.path.join(marginDir, "net_end")
        torch.save(net, netDir)
