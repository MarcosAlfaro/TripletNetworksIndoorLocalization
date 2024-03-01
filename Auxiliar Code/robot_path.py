"""THIS CODE REPRESENTS THE MAP OF A TRAINING, VALIDATION OR TEST SET
    The points represents the coordinates of the images captured by the robot:
    -blue: set images
    -red: closest images to the geometric centre of every room (representative image)
    -green: geometric centres of every room
    -yellow: first images captured in every room
    -orange: last images captured in every room
    """

import torchvision.datasets as dset
import matplotlib.pyplot as plt
import os

baseDir = os.getcwd()


def get_coords(imageDir):
    idxX = imageDir.index('_x')
    idxY = imageDir.index('_y')
    idxA = imageDir.index('_a')

    x = float(imageDir[idxX + 2:idxY])
    y = float(imageDir[idxY + 2:idxA])
    return x, y


def get_axes_limits(x, y, maxX, minX, maxY, minY):
    if x < minX:
        minX = x
    if x > maxX:
        maxX = x
    if y < minY:
        minY = y
    if y > maxY:
        maxY = y
    return maxX, minX, maxY, minY


mapX, mapY = [], []
centerXlist, centerYlist = [], []
closestXlist, closestYlist = [], []
firstXlist, firstYlist = [], []
lastXlist, lastYlist = [], []

xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000

datasetDir = os.path.join(baseDir, "DATASETS", "FRIBURGO", "Entrenamiento")
folderDataset = dset.ImageFolder(root=datasetDir)
rooms = folderDataset.classes

for room in rooms:
    roomDir = os.path.join(datasetDir, room)
    imgsList = os.listdir(roomDir)
    centerX, centerY = 0, 0

    for img in imgsList:
        imgDir = os.path.join(roomDir, img)

        coordX, coordY = get_coords(imgDir)

        xmax, xmin, ymax, ymin = get_axes_limits(coordX, coordY, xmax, xmin, ymax, ymin)

        centerX += coordX
        centerY += coordY

        if imgsList.index(img) == 0:
            firstXlist.append(coordX)
            firstYlist.append(coordY)
        elif imgsList.index(img) == len(imgsList) - 1:
            lastXlist.append(coordX)
            lastYlist.append(coordY)
        else:
            mapX.append(coordX)
            mapY.append(coordY)

    centerX /= len(imgsList)
    centerY /= len(imgsList)

    centerXlist.append(centerX)
    centerYlist.append(centerY)

imgRepDir = os.path.join(baseDir, "DATASETS", "FRIBURGO", "RepresentativeImages")

for room in rooms:
    roomDir = os.path.join(imgRepDir, room)
    imgsList = os.listdir(roomDir)

    for img in imgsList:
        imgDir = os.path.join(roomDir, img)

        coordX, coordY = get_coords(imgDir)

        closestXlist.append(coordX)
        closestYlist.append(coordY)

plt.scatter(mapX, mapY, color='blue', label="Robot path")
plt.scatter(closestXlist, closestYlist, color='red', label="Representative images")
plt.scatter(centerXlist, centerYlist, color='green', label="Geometric centers")
plt.scatter(firstXlist, firstYlist, color='yellow', label="First images")
plt.scatter(lastXlist, lastYlist, color='orange', label="Last images")

plt.title("Robot path", fontsize=20)
plt.xlabel("x (m)", fontsize=16)
plt.ylabel("y (m)", fontsize=16)
plt.legend()
plt.axis([1.05*xmin, 1.05*xmax, 1.05*ymin, 1.05*ymax])
plt.grid()
plt.savefig(os.path.join(baseDir, "FIGURES", "RobotPath.png"))
plt.show()
