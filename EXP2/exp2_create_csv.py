"""
THIS PROGRAM CREATES ALL THE CSV NEEDED TO LAUNCH THE TEST SCRIPTS IN EXPERIMENT 2
Therefore, it must be executed before any other test script of the experiment 2
Each function generates the corresponding CSV file which contains the list of test images that will be loaded into the model
exp2_generate_datasets.py script must be executed before
"""


import os
import csv
import numpy as np
from config import PARAMS
from functions import create_path, get_coords

datasetDir = os.path.join(PARAMS.datasetDir, "COLD_EFFECTS")
condIlum = ['Cloudy', 'Night', 'Sunny']

csvDir = os.path.join(PARAMS.csvDir, "EXP2")


def visual_model(effect, value):

    vmDir = os.path.join(datasetDir, effect, str(value), "Train")

    with open(csvDir + '/' + effect + '/VisualModel' + str(value) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        imgsVM, coordsVM = [], []
        rooms = sorted(os.listdir(vmDir))
        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsDir = os.listdir(roomDir)
            for image in imgsDir:
                imgDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgDir)
                imgsVM.append(imgDir)
                coordsVM.append(np.array([coordX, coordY]))
                writer.writerow([imgDir, rooms.index(room), coordX, coordY])

    return imgsVM, coordsVM


def test(ilum, effect, value):

    testDir = os.path.join(datasetDir, effect, str(value), "Test" + ilum)

    with open(csvDir + '/' + effect + '/Test' + ilum + str(value) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        rooms = sorted(os.listdir(testDir))
        for room in rooms:
            roomDir = os.path.join(testDir, room)
            imgsList = os.listdir(roomDir)
            for image in imgsList:
                imgValDir = os.path.join(roomDir, image)
                coordX, coordY = get_coords(imgValDir)
                writer.writerow([imgValDir, rooms.index(room), coordX, coordY])


def rep_images(effect, value):

    with open(csvDir + '/' + effect + '/RepImgs' + str(value) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coord X", "Coord Y"])

        repDir = create_path(os.path.join(datasetDir, effect, str(value), "RepImages"))
        rooms = sorted(os.listdir(repDir))
        for room in rooms:
            roomDir = os.path.join(repDir, room)
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

    return



occlusionValues, noiseValues, blurValues = PARAMS.occlusionValues, PARAMS.noiseValues, PARAMS.blurValues


for v in occlusionValues:
    visual_model("OCCLUSIONS", v)
    rep_images("OCCLUSIONS", v)
    for il in condIlum:
        test(il, "OCCLUSIONS", v)

for v in noiseValues:
    visual_model("NOISE", v)
    rep_images("NOISE", v)
    for il in condIlum:
        test(il, "NOISE", v)

for v in blurValues:
    visual_model("BLUR", v)
    rep_images("BLUR", v)
    for il in condIlum:
        test(il, "BLUR", v)
