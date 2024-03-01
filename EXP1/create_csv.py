"""
THIS PROGRAM CREATES ALL THE CSV NEEDED TO LAUNCH EVERY TRAINING, VALIDATION AND TEST PROGRAMMES
Therefore, it must be executed before any other training or test program
"""


import os
import csv
import torchvision.datasets as dset


baseDir = os.getcwd()
datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO")

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


def train_fine_loc(room):
    trainDir = os.path.join(datasetDir, "Train", room)

    with open(csvDir + '/Exp1TrainFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])

        imgsList = os.listdir(trainDir)
        for image in imgsList:
            imgDir = os.path.join(trainDir, image)
            coordX, coordY = get_coords(imgDir)

            writer.writerow([imgDir, coordX, coordY])
    return


def validation_coarse_loc():
    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Exp1ValidationCoarseLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img Val", "Idx Room"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgList = os.listdir(roomDir)

            for imgVal in imgList:
                imgValDir = os.path.join(roomDir, imgVal)

                writer.writerow([imgValDir, rooms.index(room)])

    return


def validation_fine_loc(room):
    trainDir = os.path.join(datasetDir, "Validation", room)

    with open(csvDir + '/Exp1ValidationFineLoc' + room + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])

        imgsList = os.listdir(trainDir)

        for image in imgsList:
            imgDir = os.path.join(trainDir, image)
            coordX, coordY = get_coords(imgDir)

            writer.writerow([imgDir, coordX, coordY])


def test_coarse_loc(ilum):
    testDir = os.path.join(datasetDir, "Test" + ilum)

    with open(csvDir + '/Exp1Test' + ilum + 'CoarseLoc' + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img Test", "Idx Room"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgList = os.listdir(roomDir)

            for imgTest in imgList:
                imgTestDir = os.path.join(roomDir, imgTest)
                writer.writerow([imgTestDir, rooms.index(room)])
    return


def test_fine_loc(ilum):

    testDir = os.path.join(datasetDir, "Test" + ilum)

    with open(csvDir + '/Exp1Test' + ilum + 'FineLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img Test", "Idx Room", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            idxRoom = rooms.index(room)
            for image in imgsList:
                imgTestDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgTestDir)
                writer.writerow([imgTestDir, idxRoom, coordX, coordY])


def visual_model_fine_loc():
    testDir = datasetDir + "/Train/"

    with open(csvDir + '/Exp1VisualModelFineLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img", "Idx Room", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            idxRoom = rooms.index(room)
            for image in imgsList:
                imgTestDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgTestDir)
                writer.writerow([imgTestDir, idxRoom, coordX, coordY])


def visual_model_global_loc():
    vmDir = datasetDir + "/Train/"

    with open(csvDir + '/Exp1VisualModelGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgDir)
                writer.writerow([imgDir, coordX, coordY])


def train_global_loc():
    testDir = datasetDir + "/Train/"

    with open(csvDir + '/Exp1TrainGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Img", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            idxRoom = rooms.index(room)
            for image in imgsList:
                imgTestDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgTestDir)
                writer.writerow([imgTestDir, idxRoom, coordX, coordY])


def validation_global_loc():
    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Exp1ValidationGlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, coordX, coordY])


def test_global_loc(ilum):
    valDir = os.path.join(datasetDir, "Test" + ilum)

    with open(csvDir + '/Exp1Test' + ilum + 'GlobalLoc.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coord X", "Coord Y"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, coordX, coordY])


validation_coarse_loc()
visual_model_fine_loc()
train_global_loc()
validation_global_loc()
visual_model_global_loc()

for r in rooms:
    train_fine_loc(r)
    validation_fine_loc(r)

for il in condIlum:
    test_coarse_loc(il)
    test_fine_loc(il)
    test_global_loc(il)
