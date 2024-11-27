"""
This script includes several functions that are often used by the other scripts
"""

import torch
import numpy as np
import os
from PIL import Image
from itertools import combinations
from sklearn.neighbors import KDTree
from config import PARAMS

device = torch.device(PARAMS.device if torch.cuda.is_available() else 'cpu')


def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return x, y


def calculate_geometric_distance(xa, ya, xb, yb):
    d = np.linalg.norm(np.array([xa, ya]) - np.array([xb, yb]))
    return d


def max_distance(points):
    max_dist = 0

    for p1, p2 in combinations(points, 2):
        dist = np.linalg.norm(p1 - p2)
        if dist > max_dist:
            max_dist = dist
    return max_dist


def get_env(room):
    if 'FR-A' in room:
        env = 0
    elif 'SA-A' in room:
        env = 1
    elif 'SA-B' in room:
        env = 2
    else:
        raise ValueError("Environment not available. Valid environments: FR_A, SA_A, SA_B")
    return env


def process_image(image, tf):
    image = Image.open(image)
    if tf is not None:
        image = tf(image)
    return image
