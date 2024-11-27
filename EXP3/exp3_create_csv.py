"""
THIS PROGRAM CREATES ALL THE CSV NEEDED TO LAUNCH EVERY TRAINING, VALIDATION AND TEST SCRIPTS IN EXPERIMENT 3
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
from config import PARAMS
from functions import get_coords, calculate_geometric_distance

datasetDir = os.path.join(PARAMS.datasetDir, "3ENVIRONMENTS")

condIlum = ['Cloudy', 'Night', 'Sunny']

csvDir = os.path.join(PARAMS.csvDir, "EXP3")


def train_coarse_loc(epochLength):
    trainDir = os.path.join(datasetDir, "Train")

    with open(csvDir + '/TrainCoarseLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        # In the coarse localization, the triplet samples are chosen in such a way that:
        # Anchor and positive images must belong to the same room
        # Anchor and negative images must belong to a different room
        rooms = sorted(os.listdir(trainDir))
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
                rPos += 0.05
                rNeg += 0.05
                imgAnc, imgPos = random.choice(imgList), random.choice(imgList)
                xAnc, yAnc = get_coords(imgAnc)
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
            rPos, rNeg = PARAMS.rPos, PARAMS.rNeg


def visual_model():

    # The visual model is built with the images of the training set
    vmDir = datasetDir + "/Train/"

    with open(csvDir + '/VisualModel.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        imgsVM, roomsVM, coordsVM = [], [], []
        rooms = sorted(os.listdir(vmDir))
        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsDir = os.listdir(roomDir)
            for image in imgsDir:
                imgDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgDir)
                imgsVM.append(imgDir)
                roomsVM.append(rooms.index(room))
                coordsVM.append(np.array([coordX, coordY]))
                writer.writerow([imgDir, room, coordX, coordY])

    return


def validation():

    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Validation.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])
        rooms = sorted(os.listdir(valDir))
        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, room, coordX, coordY])


def test(ilum):

    # The test is performed under three different lighting conditions: cloudy, night and sunny
    testDir = os.path.join(datasetDir, "Test" + ilum)

    with open(csvDir + '/Test' + ilum + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])
        rooms = sorted(os.listdir(testDir))
        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, room, coordX, coordY])


def representative_images():

    # The representative image of a room is the closest image to the geometric center of that room
    with open(csvDir + '/RepImages.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])
        trainDir = os.path.join(datasetDir, "Train")
        # repDir = create_path(os.path.join(datasetDir, "RepImages"))
        rooms = sorted(os.listdir(trainDir))
        for room in rooms:
            roomDir = os.path.join(trainDir, room)
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

            writer.writerow([imgRepDir, room, coordX, coordY])

            # destinationDir = os.path.join(repDir, room, imgList[idxCenter])
            # shutil.copy(imgRepDir, destinationDir)


validation()
representative_images()
visual_model()
train_coarse_loc(epochLength=PARAMS.epochLength_coarseLoc)
for il in condIlum:
    test(il)

rooms = sorted(os.listdir(os.path.join(datasetDir, "Train")))
for room in rooms:
    train_fine_loc(epochLength=PARAMS.epochLength_fineLoc, rPos=PARAMS.rPos, rNeg=PARAMS.rNeg, room=room)
