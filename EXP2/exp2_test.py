"""
EXPERIMENT 2:
-analyze the robustness of triplet networks and hierarchical localization against different effects:
 Gaussian noise, occlusions, motion blur

Test dataset:
Cloudy: seq2cloudy2 (2595 images), Night: seq2night2 (2707 images), Sunny: seq2sunny2 (2114 images)
* the images contain the studied effects

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in two steps:

-Coarse step: room retrieval task
    -each test image is compared with the representative image of every room
    -the closest representative descriptor indicates the retrieved room
    -metric: room retrieval accuracy (%)

-Fine step: obtain the coordinates of the robot inside the retrieved room:
    -each test image is compared with the images of the visual model of the retrieved room
    -the nearest neighbour indicates the retrieved coordinates
    -metric: geometric error (m)

"""


import os
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.neighbors import KDTree
import exp2_create_datasets
from config import PARAMS

device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = os.path.join(PARAMS.csvDir, "RESULTS")
datasetDir = os.path.join(PARAMS.datasetDir, "FRIBURGO_A")
figuresDir = os.path.join("FIGURES", "EXP1", "FineLoc")

env = "FR_A"
condIlum = ['Cloudy', 'Night', 'Sunny']


trainDir = os.path.join(datasetDir, "Train")
trainDataset = dset.ImageFolder(root=trainDir)
rooms = trainDataset.classes

savedModelsDir = os.path.join(PARAMS.modelsDir, "EXP1", "HierarchicalLoc")

sl = PARAMS.lossExp2
margin = PARAMS.marginExp2

effects = ["NOISE", "OCCLUSIONS", "BLUR"]

for effect in effects:
    if effect == "NOISE":
        values = PARAMS.noiseValues
    elif effect == "OCCLUSIONS":
        values = PARAMS.occlusionValues
    elif effect == "BLUR":
        values = PARAMS.blurValues
    else:
        continue

    for v in values:

        print(f"\n\nTest {effect}, Parameter value={v}")

        imgRepDataset = exp2_create_datasets.RepImages(effect=effect, value=v)
        imgRepDataloader = DataLoader(imgRepDataset, num_workers=0, batch_size=1, shuffle=False)

        vmDataset = exp2_create_datasets.VisualModel(effect=effect, value=v)
        vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

        marginDir_CL = os.path.join(savedModelsDir, "CoarseLoc", sl, str(margin))
        testNetCL = os.path.join(marginDir_CL, os.listdir(marginDir_CL)[0])
        netCL = torch.load(testNetCL)

        lossDir_FL = os.path.join(savedModelsDir, "FineLoc", sl, str(margin))
        bestErrorRooms = 100*np.ones((len(rooms), 1))
        bestNets = np.zeros((len(rooms), 1))

        netsFL = []
        for room in rooms:
            netDir = os.path.join(lossDir_FL, room)
            netsFL.append(torch.load(os.path.join(netDir, os.listdir(netDir)[0])).to(device))

        """REPRESENTATIVE IMAGES"""

        descImgRep = []
        for i, imgRepData in enumerate(imgRepDataloader, 0):
            imgRep = imgRepData[0].to(device)
            output = netCL(imgRep).cpu().detach().numpy()[0]
            descImgRep.append(output)
        treeImgRep = KDTree(descImgRep, leaf_size=2)

        """VISUAL MODEL"""

        coordsVM, coordsVMrooms, descriptorsVM, treeDescVMrooms = [], [], [], []

        for room in rooms:
            idxRoom = rooms.index(room)
            descVMroom, coordsVMroom = [], []

            for i, VMdata in enumerate(vmDataloader, 0):
                imgVM, ind_gt, coordsImgVM = VMdata
                imgVM = imgVM.to(device)

                if ind_gt.detach().numpy()[0] == idxRoom:
                    output = netsFL[idxRoom](imgVM).cpu().detach().numpy()[0]
                    descVMroom.append(output)
                    coordsImgVM = coordsImgVM.detach().numpy()[0]
                    coordsVMroom.append(coordsImgVM)
                    coordsVM.append(coordsImgVM)
            coordsVMrooms.append(coordsVMroom)
            descriptorsVM.append(descVMroom)
            treeDescVMrooms.append(KDTree(descVMroom, leaf_size=2))

        treeCoordsVM = KDTree(coordsVM, leaf_size=2)
        """
    
    
    
    
    
    
        """

        recallLF = np.zeros((len(condIlum) + 1, PARAMS.kMax))
        accuracyCoarseLoc = np.zeros(len(condIlum) + 1)
        geomError, minErrorPossible = np.zeros(len(condIlum) + 1), np.zeros(len(condIlum) + 1)
        geomErrorRooms = np.zeros((len(condIlum) + 1, len(rooms)))
        minErrorRooms = np.zeros((len(condIlum) + 1, len(rooms)))
        for ilum in condIlum:
            idxIlum = condIlum.index(ilum)
            print(f"Test {ilum}\n")

            testDataset = exp2_create_datasets.Test(illumination=ilum, effect=effect, value=v)
            testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

            coordsMapTest = []

            for i, data in enumerate(testDataloader, 0):
                imgTest, actualRoom, coordsImgTest = data
                imgTestLg = imgTest.to(device)

                """COARSE LOCALIZATION"""

                output = netCL(imgTestLg).cpu().detach().numpy()[0]

                if sl in ['CL', 'AL']:
                    cosSimilarities = np.dot(descImgRep, output)
                    predictedRoom = np.argmax(cosSimilarities)
                else:
                    distances, predictedRoom = treeImgRep.query(output.reshape(1, -1), k=9)
                    predictedRoom = predictedRoom[0][0]

                actualRoom = actualRoom.detach().numpy()[0]
                if predictedRoom == actualRoom:
                    accuracyCoarseLoc[idxIlum] += 1

                """FINE LOCALIZATION"""
                imgTestLf = imgTest.to(device)
                output = netsFL[predictedRoom](imgTestLf).cpu().detach().numpy()[0]

                if sl in ['CL', 'AL']:
                    cosSimilarities = np.dot(descriptorsVM[predictedRoom], output)
                    idxMinPred = np.argmax(cosSimilarities)
                else:
                    treeDescVM = treeDescVMrooms[predictedRoom]
                    _, idxDesc = treeDescVM.query(output.reshape(1, -1), k=1)
                    idxMinPred = idxDesc[0][0]

                coordsImgTest = coordsImgTest.detach().numpy()[0]
                _, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=PARAMS.kMax)
                idxMinReal = idxGeom[0][0]

                coordsPredictedImg, coordsClosestImg = coordsVMrooms[predictedRoom][idxMinPred], coordsVM[idxMinReal]

                geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                geomErrorRooms[idxIlum][actualRoom] += np.linalg.norm(coordsImgTest - coordsPredictedImg)

            for room in range(len(rooms)):
                auxDir = os.path.join(datasetDir, "Test" + ilum, rooms[room])
                geomErrorRooms[idxIlum][room] /= len(os.listdir(auxDir))

            accuracyCoarseLoc[idxIlum] *= 100 / len(testDataloader)
            geomError[idxIlum] /= len(testDataloader)


            print(f"COARSE LOC\nAccuracy: {accuracyCoarseLoc[idxIlum]} %\n")
            print(f"FINE LOC")

            print(f"Average {ilum} Error: {geomError[idxIlum]} m")
            print("\n")

        accuracyCoarseLoc[-1] = np.average(accuracyCoarseLoc[0:-1])
        geomError[-1] = np.average(geomError[0:-1])
        geomErrorRooms[-1] = np.average(geomErrorRooms[0:-1], axis=0)

        print(f"Average Error: {geomError[-1]} m\n")

        for room in range(len(rooms)):
            if geomErrorRooms[-1][room] < bestErrorRooms[room]:
                bestErrorRooms[room] = geomErrorRooms[-1][room]
