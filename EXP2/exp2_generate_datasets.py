"""
THIS PROGRAM IS USED TO GENERATE THE IMAGE SETS WITH THE DIFFERENT EFFECTS APPLIED ON THE IMAGES
The effects employed in this experiment are: Gaussian noise, occlusions and motion blur
Therefore, it must be executed before any other training or test program of EXP2

YAML PARAMETERS TO TAKE INTO ACCOUNT:
Parameter values for each effect: occluded columns (occlusionValues), sigma (noiseValues) and kernel size (blurValues)
"""


import os
import random
import numpy as np
from config import PARAMS
from PIL import Image, ImageDraw, ImageFilter
from functions import create_path


datasetDir = os.path.join(PARAMS.datasetDir, "FRIBURGO_A")
newDatasetDir = create_path(os.path.join(PARAMS.datasetDir, "COLD_EFFECTS"))
# imgRepDir = create_path(os.path.join(newDatasetDir, "RepImages"))


sets = ["Train", "RepImages", "Validation", "TestCloudy", "TestNight", "TestSunny"]


# The occlusion effect is applied by setting some columns of the image as black (pixels will take a zero value)
# Parameter to set: number of columns occluded
def apply_occlusions(img, x):

    width, height = img.size

    occluded_width = x
    occluded_height = height - 1

    x1 = random.randint(0, width - occluded_width)
    y1 = 0

    x2 = x1 + occluded_width
    y2 = y1 + occluded_height

    draw = ImageDraw.Draw(img)

    if x2 >= width:
        draw.rectangle([x1, y1, width-1, y2], fill=(0, 0, 0))
        draw.rectangle([0, y1, x2 - width, y2], fill=(0, 0, 0))
    else:
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

    return img


# The occlusion effect is applied by a convolution mask that blurs the image in the horizontal axis
# Parameter to set: kernel size
def apply_motion_blur(imagen, kernel_size):

    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be an odd and positive number.")

    blurred_image = imagen.filter(ImageFilter.BoxBlur(kernel_size // 2))

    return blurred_image


# The noise effect is applied randomly on the pixels following a Gaussian distribution
#  Parameter to set: sigma
def apply_noise(img, sigma):

    img = np.array(img)
    mean = 0
    noise = np.random.normal(mean, sigma, img.shape)
    noise_image = img + noise
    noise_image = np.clip(noise_image, 0, 255).astype(np.uint8)
    noise_image = Image.fromarray(noise_image)

    return noise_image


occlusionValues = PARAMS.occlusionValues
for size in occlusionValues:
    sizeDir = create_path(os.path.join(newDatasetDir, "OCCLUSIONS", str(size)))
    for imageSet in sets:
        setDir = os.path.join(datasetDir, imageSet)
        rooms = os.listdir(setDir)
        newSetDir =  create_path(os.path.join(sizeDir, imageSet))
        for room in rooms:
            roomDir = os.path.join(setDir, room)
            imgList = os.listdir(roomDir)
            newRoomDir = create_path(os.path.join(newSetDir, room))
            for img in imgList:
                imgDir = os.path.join(roomDir, img)
                newImgDir = os.path.join(newRoomDir, img)
                image = Image.open(imgDir)
                newImage = apply_occlusions(image, size)
                newImage.save(newImgDir)


blurValues = PARAMS.blurValues
for s in blurValues:
    sigmaDir = create_path(os.path.join(newDatasetDir, "BLUR", str(s)))
    for imageSet in sets:
        setDir = os.path.join(datasetDir, imageSet)
        rooms = os.listdir(setDir)
        newSetDir =  create_path(os.path.join(sigmaDir, imageSet))
        for room in rooms:
            roomDir = os.path.join(setDir, room)
            imgList = os.listdir(roomDir)
            newRoomDir = create_path(os.path.join(newSetDir, room))
            for img in imgList:
                imgDir = os.path.join(roomDir, img)
                newImgDir = os.path.join(newRoomDir, img)
                image = Image.open(imgDir)
                newImage = apply_motion_blur(image, s)
                newImage.save(newImgDir)


noiseValues = PARAMS.noiseValues
for s in noiseValues:
    sigmaDir = create_path(os.path.join(newDatasetDir, "NOISE", str(s)))
    for imageSet in sets:
        setDir = os.path.join(datasetDir, imageSet)
        rooms = os.listdir(setDir)
        newSetDir =  create_path(os.path.join(sigmaDir, imageSet))
        for room in rooms:
            roomDir = os.path.join(setDir, room)
            imgList = os.listdir(roomDir)
            newRoomDir = create_path(os.path.join(newSetDir, room))
            for img in imgList:
                imgDir = os.path.join(roomDir, img)
                newImgDir = os.path.join(newRoomDir, img)
                image = Image.open(imgDir)
                newImage = apply_noise(image, s)
                newImage.save(newImgDir)
