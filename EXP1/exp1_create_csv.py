"""
THIS PROGRAM CREATES ALL THE CSV NEEDED TO LAUNCH EVERY TRAINING, VALIDATION AND TEST SCRIPTS IN EXPERIMENT 1
Therefore, it must be executed before any other training or test script
Each function generates the corresponding CSV file which contains the list of inputs that will be loaded into the model

YAML PARAMETERS TO TAKE INTO ACCOUNT:
Directories: datasetDir, csvDir
Epoch length: epochLength_coarseLoc, epochLength_fineLoc, epochLength_globalLoc
Threshold distances: rPos, rNeg
"""


import os
import csv
import random
import numpy as np
from sklearn.neighbors import KDTree
from functions import get_coords, calculate_geometric_distance, max_distance
import torchvision.datasets as dset
from config import PARAMS


datasetDir = os.path.join(PARAMS.datasetDir, "FRIBURGO_A")

trainDataset = dset.ImageFolder(root=datasetDir + "/Train/")
rooms = trainDataset.classes

condIlum = ['Cloudy', 'Night', 'Sunny']

csvDir = os.path.join(PARAMS.csvDir, "EXP1")


def train_coarse_loc(epochLength):
    trainDir = os.path.join(datasetDir, "Train")

    with open(csvDir + '/TrainCoarseLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        # In the coarse localization, the triplet samples are chosen in such a way that:
        # Anchor and positive images must belong to the same room
        # Anchor and negative images must belong to a different room
        for _ in range(epochLength):
            roomAnc = random.choice(rooms)
            roomAncDir = os.path.join(trainDir, roomAnc)
            imgList = os.listdir(roomAncDir)
            imgAnc, imgPos = random.choice(imgList), random.choice(imgList)
            while imgAnc == imgPos:
                imgPos = random.choice(imgList)
            roomNeg = random.choice(rooms)
            while roomAnc == roomNeg:
                roomNeg = random.choice(rooms)
            roomNegDir = os.path.join(trainDir, roomNeg)
            imgList = os.listdir(roomNegDir)
            imgNeg = random.choice(imgList)

            imgAncDir, imgPosDir, imgNegDir =\
                os.path.join(roomAncDir, imgAnc), os.path.join(roomAncDir, imgPos), os.path.join(roomNegDir, imgNeg)

            writer.writerow([imgAncDir, imgPosDir, imgNegDir])


def train_fine_loc(epochLength, room, rPos, rNeg):
    trainDir = os.path.join(datasetDir, "Train")

    with open(csvDir + '/TrainFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        # In the coarse localization, the triplet samples are chosen in such a way that:
        # Anchor, positive and negative images must belong to the same room
        # The distance between anchor and positive images must be lower than a threshold r+
        # The distance between anchor and negative images must be larger than a threshold r-
        # r+ and r- must be defined in YAML file, please note that r+ <= r-
        for _ in range(epochLength):

            roomDir = os.path.join(trainDir, room)
            imgList = os.listdir(roomDir)
            imgAnc, imgPos = random.choice(imgList), random.choice(imgList)
            xAnc, yAnc = get_coords(imgAnc)
            xPos, yPos = get_coords(imgPos)
            dist = calculate_geometric_distance(xAnc, yAnc, xPos, yPos)
            while imgAnc == imgPos or dist > rPos:
                imgPos = random.choice(imgList)
                xPos, yPos = get_coords(imgPos)
                dist = calculate_geometric_distance(xAnc, yAnc, xPos, yPos)
            imgNeg = random.choice(imgList)
            xNeg, yNeg = get_coords(imgNeg)
            dist = calculate_geometric_distance(xAnc, yAnc, xNeg, yNeg)
            while dist < rNeg:
                imgNeg = random.choice(imgList)
                xNeg, yNeg = get_coords(imgNeg)
                dist = calculate_geometric_distance(xAnc, yAnc, xNeg, yNeg)

            imgAnc, imgPos, imgNeg = \
                os.path.join(roomDir, imgAnc), os.path.join(roomDir, imgPos), os.path.join(roomDir, imgNeg)

            writer.writerow([imgAnc, imgPos, imgNeg])


def train_global_loc(epochLength, tree, rPos, rNeg):

    with open(csvDir + '/TrainGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        # In the coarse localization, the triplet samples are chosen in such a way that:
        # The distance between anchor and positive images must be lower than a threshold r+
        # The distance between anchor and negative images must be larger than a threshold r-
        # r+ and r- must be defined in YAML file, please note that r+ <= r-
        for _ in range(epochLength):

            idxAnc = random.randrange(len(imgsList))
            imgAnc, coordsAnc = imgsList[idxAnc], coordsList[idxAnc]

            indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rPos)[0]
            idxPos = random.choice(indices)
            while idxAnc == idxPos:
                idxPos = random.choice(indices)
            imgPos = imgsList[idxPos]

            indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rNeg)[0]
            idxNeg = random.randrange(len(imgsList))
            while idxNeg in indices or idxAnc == idxNeg:
                idxNeg = random.randrange(len(imgsList))
            imgNeg = imgsList[idxNeg]


            writer.writerow([imgAnc, imgPos, imgNeg])
    return


def visual_model():

    # The visual model is built with the images of the training set
    vmDir = datasetDir + "/Train/"

    with open(csvDir + '/VisualModel.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        imgsVM, roomsVM, coordsVM = [], [], []
        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsDir = os.listdir(roomDir)
            for image in imgsDir:
                imgDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgDir)
                imgsVM.append(imgDir)
                roomsVM.append(rooms.index(room))
                coordsVM.append(np.array([coordX, coordY]))
                writer.writerow([imgDir, rooms.index(room), coordX, coordY])

    return imgsVM, roomsVM, coordsVM


def validation():

    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Validation.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, rooms.index(room), coordX, coordY])


def test(ilum):

    # The test is performed under three different lighting conditions: cloudy, night and sunny
    testDir = os.path.join(datasetDir, "Test" + ilum)

    with open(csvDir + '/Test' + ilum + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, rooms.index(room), coordX, coordY])


def representative_images():

    # The representative image of a room is the closest image to the geometric center of that room
    with open(csvDir + '/RepImages.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        # repDir = create_path(os.path.join(datasetDir, "RepImages"))

        for room in rooms:
            roomDir = os.path.join(datasetDir, "Train", room)
            imgList = os.listdir(roomDir)
            centerCoords = np.array([0.0, 0.0])

            for img in imgList:
                imgDir = os.path.join(roomDir, img)
                coordX, coordY = get_coords(imgDir)
                centerCoords += np.array([coordX, coordY])
            centerCoords /= len(imgList)

            distances = []
            for img in imgList:
                imgDir = os.path.join(roomDir, img)
                coordX, coordY = get_coords(imgDir)
                distances.append(np.linalg.norm(np.array([coordX, coordY]) - centerCoords))

            minDist, idxCenter = min(distances), np.argmin(distances)
            imgRepDir = os.path.join(roomDir, imgList[idxCenter])
            coordX, coordY = get_coords(imgList[idxCenter])

            writer.writerow([imgRepDir, rooms.index(room), coordX, coordY])

            # destinationDir = os.path.join(repDir, room, imgList[idxCenter])
            # shutil.copy(imgRepDir, destinationDir)


# This function is for siamese networks (only for comparison)
def snn_train_coarse_loc(epochLength, same_probability):
    trainDir = os.path.join(datasetDir, "Train")

    with open(csvDir + '/SNNTrainCoarseLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img0", "Img1", "Label"])

        for _ in range(epochLength):
            room0 = random.choice(rooms)
            room0Dir = os.path.join(trainDir, room0)
            imgList = os.listdir(room0Dir)
            img0 = random.choice(imgList)
            numberList = [0, 1]
            sameRoom = np.random.choice(numberList, 1, p=[same_probability, 1 - same_probability])[0]

            if not sameRoom:
                img1 = random.choice(imgList)
                while img0 == img1:
                    img1 = random.choice(imgList)
                room1Dir = room0Dir
            else:
                room1 = random.choice(rooms)
                while room0 == room1:
                    room1 = random.choice(rooms)
                room1Dir = os.path.join(trainDir, room1)
                imgList = os.listdir(room1Dir)
                img1 = random.choice(imgList)

            img0Dir, img1Dir = os.path.join(room0Dir, img0), os.path.join(room1Dir, img1)

            writer.writerow([img0Dir, img1Dir, sameRoom])


# This function is for siamese networks (only for comparison)
def snn_train_global_loc(epochLength):

    with open(csvDir + '/SNNTrainGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img0", "Img1", "Label"])

        maxDist = max_distance(coordsList)
        for i in range(epochLength):
            idx0, idx1 = random.randrange(len(imgsList)), random.randrange(len(imgsList))
            while idx0 == idx1:
                idx0, idx1 = random.randrange(len(imgsList)), random.randrange(len(imgsList))
            img0, room0, coords0 = imgsList[idx0], roomsList[idx0], coordsList[idx0]
            img1, room1, coords1 = imgsList[idx1], roomsList[idx1], coordsList[idx1]
            dist = np.linalg.norm(coords1 - coords0)
            label = dist/maxDist
            writer.writerow([img0, img1, label])
    return


# This function is for siamese networks (only for comparison)
def snn_train_fine_loc(epochLength, room):
    trainDir = os.path.join(datasetDir, "Train")

    with open(csvDir + '/SNNTrainFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img0", "Img1", "Label"])

        roomDir = os.path.join(trainDir, room)
        imgList = os.listdir(roomDir)
        coordsRoomList =  []
        for i in range(len(imgList)):
            imgX, imgY = get_coords(imgList[i])
            coordsRoomList.append(np.array([imgX, imgY]))
        maxRoomDist = max_distance(coordsRoomList)

        for _ in range(epochLength):

            img0, img1 = random.choice(imgList), random.choice(imgList)
            while img0 == img1:
                img0, img1 = random.choice(imgList), random.choice(imgList)
            xAnc, yAnc = get_coords(img0)
            xPos, yPos = get_coords(img1)
            dist = calculate_geometric_distance(xAnc, yAnc, xPos, yPos)
            label = dist/maxRoomDist

            img0Dir, img1Dir = os.path.join(roomDir, img0), os.path.join(roomDir, img1)

            writer.writerow([img0Dir, img1Dir, label])


validation()
representative_images()
imgsList, roomsList, coordsList = visual_model()
for il in condIlum:
    test(il)
treeVM = KDTree(coordsList, leaf_size=2)
train_coarse_loc(epochLength=PARAMS.epochLength_coarseLoc)
train_global_loc(epochLength=PARAMS.epochLength_globalLoc, tree=treeVM, rPos=PARAMS.rPos, rNeg=PARAMS.rNeg)
snn_train_coarse_loc(epochLength=PARAMS.epochLength_coarseLoc, same_probability=PARAMS.sameP)
snn_train_global_loc(epochLength=PARAMS.epochLength_globalLoc)

for room in rooms:
    train_fine_loc(epochLength=PARAMS.epochLength_fineLoc, rPos=PARAMS.rPos, rNeg=PARAMS.rNeg, room=room)
    snn_train_fine_loc(epochLength=PARAMS.epochLength_fineLoc, room=room)
