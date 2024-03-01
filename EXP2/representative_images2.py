"""
THIS PROGRAM OBTAINS THE REPRESENTATIVE IMAGE OF EACH ROOM
The representative image of a room is the closest image to the geometric centre of that room
If the folder "RepresentativeImage" does not exist, please uncomment the commented lines and execute this program
"""


import torchvision.datasets as dset
import os
import csv
import shutil
from config import PARAMETERS


def get_coords(imageDir):
    idxX = imageDir.index('_x')
    idxY = imageDir.index('_y')
    idxA = imageDir.index('_a')

    x = float(imageDir[idxX + 2:idxY])
    y = float(imageDir[idxY + 2:idxA])
    return x, y


baseDir = os.getcwd()
csvDir = os.path.join(baseDir, "CSV")
datasetDir = os.path.join(baseDir, "DATASETS", "3ENVIRONMENTS")
trainingDir = os.path.join(datasetDir, "Train")

trainingDataset = dset.ImageFolder(root=trainingDir)
rooms = trainingDataset.classes


with open(csvDir + '/Exp2RepresentativeImages.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Room", "Idx Room", "Image", "Idx Image", "CoordX", "CoordY"])

    idxRoom = 0
    os.mkdir(os.path.join(datasetDir, "RepresentativeImages"))
    for room in rooms:
        os.mkdir(os.path.join(datasetDir, "RepresentativeImages", rooms[idxRoom]))
        roomDir = os.path.join(trainingDir, room)
        imgList = os.listdir(roomDir)
        centerX, centerY = 0, 0

        for img in imgList:
            imgDir = os.path.join(roomDir, img)
            coordX, coordY = get_coords(imgDir)

            centerX += coordX
            centerY += coordY

        centerX /= len(imgList)
        centerY /= len(imgList)

        minDist = 1000000
        idxCenter = 0
        idxImg = 0

        for img in imgList:
            imgDir = os.path.join(roomDir, img)
            coordX, coordY = get_coords(imgDir)

            dist = ((coordX-centerX)**2+(coordY-centerY)**2)**0.5

            if dist < minDist:
                minDist = dist
                idxCenter = idxImg
                centerDir = imgDir
                imgCenterX = coordX
                imgCenterY = coordY

            idxImg += 1

        writer.writerow([roomDir, idxRoom, centerDir, idxCenter, imgCenterX, imgCenterY])
        destinationDir = os.path.join(datasetDir, "RepresentativeImages", rooms[idxRoom])
        shutil.copy(centerDir, destinationDir)

        print(f"Room: {rooms[idxRoom]}")
        print(f"Room geometric center: ({centerX}, {centerY}) m")
        print(f"Closest image: ({imgCenterX}, {imgCenterY}) m, {centerDir}")
        print(f"Distance to geometric center: {minDist} m")

        idxRoom += 1
