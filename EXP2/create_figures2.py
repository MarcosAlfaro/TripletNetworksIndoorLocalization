"""
THIS PROGRAM CONTAINS FUNCTIONS THAT CREATE FIGURES THAT GRAPHICALLY REPRESENT THE TEST RESULTS OF A LOCALIZATION STAGE
Among these graphics, we can find:
- Confusion matrices (coarse step)
- Maps with the network predictions (fine and global steps)
- Graphics that show the errors made by the network in every room (fine step)
Test programmes will call these functions
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
# import matplotlib.patches as patches
import os
import numpy as np

from config import PARAMETERS


def display_confusion_matrix_room(actual, predicted, plt_name, rooms, loss, ilum):

    cm = confusion_matrix(actual, predicted)
    print(f"Confusion matrix: {ilum}")
    print(cm)
    plt.figure(figsize=(15, 15), dpi=120)
    df_cm = pd.DataFrame(cm, index=rooms, columns=rooms)

    sn.set(font_scale=1.3)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='d', cbar=False)  # font size

    plt.suptitle(f'Coarse localization, Room retrieval', fontsize=24)
    plt.title(f"Loss function: {loss}, Illumination condition: {ilum}", fontsize=20)
    plt.ylabel('Actual room', fontsize=18)
    plt.xlabel('Predicted room', fontsize=18)
    plt.savefig(plt_name, dpi=400)
    plt.close()


def display_confusion_matrix_env(actual, predicted, plt_name, rooms, loss, ilum):

    cm = confusion_matrix(actual, predicted)
    print(f"Confusion matrix: {ilum}")
    print(cm)
    plt.figure(figsize=(9, 9), dpi=120)
    if ilum == "Sunny":
        df_cm = pd.DataFrame(cm, index=['FR-A', 'SA-B'], columns=['FR-A', 'SA-B'])
    else:
        df_cm = pd.DataFrame(cm, index=['FR-A', 'SA-A', 'SA-B'], columns=['FR-A', 'SA-A', 'SA-B'])

    sn.set(font_scale=1.3)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Blues', fmt='d', cbar=False)  # font size

    plt.suptitle(f'Coarse localization, Environment retrieval', fontsize=24)
    plt.title(f"Loss function: {loss}, Illumination condition: {ilum}", fontsize=20)
    plt.ylabel('Actual environment', fontsize=18)
    plt.xlabel('Predicted environment', fontsize=18)
    plt.savefig(plt_name, dpi=400)
    plt.close()


def get_axes_limits(coordX, coordY, xmax, xmin, ymax, ymin):
    if coordX < xmin:
        xmin = coordX
    if coordX > xmax:
        xmax = coordX
    if coordY < ymin:
        ymin = coordY
    if coordY > ymax:
        ymax = coordY
    return xmax, xmin, ymax, ymin


def display_coord_map(direc, stage, mapVM, mapTest, k, ilum, loss):

    sl = loss
    idxLoss = PARAMETERS.lossAbreviations.index(sl)
    loss = PARAMETERS.losses[idxLoss]

    xmin, xmax, ymin, ymax = 1000, -1000, 1000, -1000
    plt.figure(figsize=(9, 5), dpi=120, edgecolor='black')

    firstk1, firstErrork, firstErrorRoom = True, True, True

    for vm in range(len(mapVM)):
        if vm == 0:
            plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue', label="Visual Model")
        else:
            plt.scatter(mapVM[vm][0], mapVM[vm][1], color='blue')
        xmax, xmin, ymax, ymin = get_axes_limits(mapVM[vm][0], mapVM[vm][1], xmax, xmin, ymax, ymin)

    for t in range(len(mapTest)):
        if mapTest[t][4] == "R":
            if firstErrorRoom:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='brown', label="Wrong Room")
                firstErrorRoom = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='brown')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], 'brown')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        elif mapTest[t][4] == "F":
            if firstErrork:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='red', label="Error")
                firstErrork = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='red')
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], 'red')
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)
        else:
            label = int(mapTest[t][4])
            if label < k / 2:
                color = (2 * label / k, 1, 0)
            elif label > k / 2:
                color = (1, 1 - 2*(label - k/2) / k, 0)
            else:
                color = (1, 1, 0)
            if firstk1:
                plt.scatter(mapTest[t][2], mapTest[t][3], color='green', label='Prediction amongst K=1')
                firstk1 = False
            else:
                plt.scatter(mapTest[t][2], mapTest[t][3], color=color)
                plt.plot([mapTest[t][0], mapTest[t][2]], [mapTest[t][1], mapTest[t][3]], color=color)
            xmax, xmin, ymax, ymin = get_axes_limits(mapTest[t][2], mapTest[t][3], xmax, xmin, ymax, ymin)

    plt.axis([xmin-0.5, xmax+0.5, ymin-0.25, ymax+0.25])
    plt.ylabel('y (m)')
    plt.xlabel('x (m)')
    plt.suptitle(stage + ' localization')
    plt.title(f'Loss function: {loss}, Illumination: {ilum}')
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(direc, "map_" + sl + "_" + ilum + ".png"), dpi=400)
    # plt.show()
    plt.close()

    return


def error_rooms(direc, geomError, minError, geomErrorRooms, minErrorRooms, rooms, loss):
    plt.figure(figsize=(9, 5), dpi=120, edgecolor='black')

    """MUST ADD MANUALLY ROOM LABELS IN X AXIS"""
    condIlum = ['Cloudy', 'Night', 'Sunny']
    colors = ['blue', 'black', 'red']
    maxError = np.max(geomErrorRooms)
    idxLoss = PARAMETERS.losses.index(loss)
    sl = PARAMETERS.lossAbreviations[idxLoss]
    for ilum in condIlum:
        idxIlum = condIlum.index(ilum)
        x = 0.25*idxIlum - 0.25
        color = colors[idxIlum]

        plt.scatter(1 + x, geomErrorRooms[idxIlum][0], color=color, label=ilum + " error", marker="x")
        plt.scatter(1 + x, minErrorRooms[idxIlum][0], color=color, label=ilum + " minimum error", marker="_")
        plt.plot([1+x, 1+x], [geomErrorRooms[idxIlum][0], minErrorRooms[idxIlum][0]], color)

        for i in range(1, len(rooms)):
            if not (ilum == "Sunny" and 'SA-A' in rooms[i]):
                plt.scatter(i + 1 + x, geomErrorRooms[idxIlum][i], color=color, marker="x")
                plt.scatter(i + 1 + x, minErrorRooms[idxIlum][i], color=color, marker="_")
                plt.plot([i + 1 + x, i + 1 + x], [geomErrorRooms[idxIlum][i], minErrorRooms[idxIlum][i]], color)

        plt.scatter(len(rooms) + 1 + x, geomError[idxIlum], color=color, marker="x")
        plt.scatter(len(rooms) + 1 + x, minError[idxIlum], color=color, marker="_")
        plt.plot([len(rooms) + 1 + x, len(rooms) + 1 + x], [geomError[idxIlum], minError[idxIlum]], color)

    plt.grid()
    plt.axis([0, len(rooms) + 2, 0, maxError + 0.1])
    plt.ylabel('Geometric error (m)')
    plt.xlabel('Rooms')
    plt.title(f'Hierarchical Localization, Loss function: {loss}')
    plt.suptitle('Geometric Error Per Room')
    plt.legend(fontsize=13)
    plt.savefig(os.path.join(direc, "RoomErrors" + sl + ".png"))
    plt.close()

    return
