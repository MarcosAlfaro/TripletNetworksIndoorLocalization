"""
THIS PROGRAM CREATES ALL THE CSV NEEDED TO LAUNCH EVERY TRAINING, VALIDATION AND TEST PROGRAMMES
Therefore, it must be executed before any other training or test program
"""


import os
import csv
import torchvision.datasets as dset

from config import PARAMETERS

baseDir = os.getcwd()
datasetDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS")

trainDataset = dset.ImageFolder(root=datasetDir + "/Train/")
rooms = trainDataset.classes

condIlum = ['Cloudy', 'Night', 'Sunny']

# os.mkdir(os.path.join(baseDir, "CSV"))
csvDir = os.path.join(baseDir, "CSV")


def get_coords(imageDir):
    idxX = imageDir.index('_x')
    idxY = imageDir.index('_y')
    idxA = imageDir.index('_a')

    x = float(imageDir[idxX + 2:idxY])
    y = float(imageDir[idxY + 2:idxA])
    return x, y


def get_env(room):
    if 'FR-A' in room:
        env = 0
    elif 'SA-A' in room:
        env = 1
    else:
        env = 2
    return env


def validation_coarse_loc():
    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Exp2ValidationCoarseLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img Val", "Idx Env", "Idx Room"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgList = os.listdir(roomDir)
            idxEnv = get_env(room)

            for imgVal in imgList:
                imgValDir = os.path.join(roomDir, imgVal)
                writer.writerow([imgValDir, idxEnv, rooms.index(room)])
    return


def test_coarse_loc(ilum):
    testDir = os.path.join(datasetDir, "Test" + ilum)
    testDataset = dset.ImageFolder(root=testDir)
    roomsTest = testDataset.classes

    with open(csvDir + '/Exp2Test' + ilum + 'CoarseLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img Test", "Idx Env", "Idx Room"])
        for room in roomsTest:
            roomDir = os.path.join(testDir, room)
            imgList = os.listdir(roomDir)
            idxEnv = get_env(room)
            for imgTest in imgList:
                imgTestDir = os.path.join(roomDir, imgTest)
                if ilum == "Sunny" and 'SA-B' in room:
                    writer.writerow([imgTestDir, idxEnv, roomsTest.index(room)+8])
                else:
                    writer.writerow([imgTestDir, idxEnv, roomsTest.index(room)])
    return


def train_fine_loc(room):
    trainDir = os.path.join(datasetDir, "Train", room)

    with open(csvDir + '/Exp2TrainFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])
        imgsList = os.listdir(trainDir)
        for image in imgsList:
            imgDir = os.path.join(trainDir, image)
            coordX, coordY = get_coords(imgDir)

            writer.writerow([imgDir, coordX, coordY])
    return


def visual_model_train_fine_loc(room):
    trainDir = os.path.join(datasetDir, "Train", room)

    with open(csvDir + '/Exp2VisualModelTrainFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])

        imgsList = os.listdir(trainDir)

        for image in imgsList:
            imgDir = os.path.join(trainDir, image)
            coordX, coordY = get_coords(imgDir)

            writer.writerow([imgDir, coordX, coordY])
    return


def validation_fine_loc(room):
    trainDir = os.path.join(datasetDir, "AugmentedSets", "Validation", room)

    with open(csvDir + '/Exp2ValidationFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])

        imgsList = os.listdir(trainDir)

        for image in imgsList:
            imgDir = os.path.join(trainDir, image)
            coordX, coordY = get_coords(imgDir)

            writer.writerow([imgDir, coordX, coordY])


def test_fine_loc(ilum):
    testDir = os.path.join(datasetDir, "Test" + ilum)
    testDataset = dset.ImageFolder(root=testDir)
    roomsTest = testDataset.classes

    with open(csvDir + '/Exp2Test' + ilum + 'FineLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img Test", "Idx Env", "Idx Room", "Coord X", "Coord Y"])

        for room in roomsTest:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            if ilum == "Sunny" and "SA-B" in room:
                idxRoom = roomsTest.index(room) + 8
            else:
                idxRoom = roomsTest.index(room)
            for image in imgsList:
                imgTestDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgTestDir)
                idxEnv = get_env(room)

                writer.writerow([imgTestDir, idxEnv, idxRoom, coordX, coordY])


def visual_model_test_fine_loc():
    testDir = datasetDir + "/Train/"

    with open(csvDir + '/Exp2VisualModelTestFineLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img", "Idx Env", "Idx Room", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            idxRoom = rooms.index(room)
            idxEnv = get_env(room)
            for image in imgsList:
                imgTestDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgTestDir)
                writer.writerow([imgTestDir, idxEnv, idxRoom, coordX, coordY])


def train_global_loc():
    testDir = datasetDir + "/Train/"

    with open(csvDir + '/Exp2TrainGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img", "IdxEnv", "IdxRoom", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            idxRoom = rooms.index(room)
            idxEnv = get_env(room)
            for image in imgsList:
                imgTestDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgTestDir)
                writer.writerow([imgTestDir, idxEnv, idxRoom, coordX, coordY])


def visual_model_global_loc():
    vmDir = datasetDir + "/Train/"

    with open(csvDir + '/Exp2VisualModelGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Env", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsList = os.listdir(roomDir)
            idxEnv = get_env(room)
            for image in imgsList:
                imgDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgDir)
                writer.writerow([imgDir, idxEnv, coordX, coordY])


def validation_global_loc():
    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Exp2ValidationGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Env", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsList = os.listdir(roomDir)
            idxEnv = get_env(room)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, idxEnv, coordX, coordY])


def test_global_loc(ilum):
    testDir = os.path.join(datasetDir, "Test" + ilum)
    testDataset = dset.ImageFolder(root=testDir)
    roomsTest = testDataset.classes

    with open(csvDir + '/Exp2Test' + ilum + 'GlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Env", "Coord X", "Coord Y"])

        for room in roomsTest:
            roomDir = os.path.join(testDir, room)
            idxEnv = get_env(room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, idxEnv, coordX, coordY])


validation_coarse_loc()
visual_model_test_fine_loc()
validation_global_loc()
visual_model_global_loc()
train_global_loc()

for r in rooms:
    visual_model_train_fine_loc(r)
    validation_fine_loc(r)
    train_fine_loc(r)
    validation_fine_loc(r)

for il in condIlum:
    test_coarse_loc(il)
    test_fine_loc(il)
    test_global_loc(il)
