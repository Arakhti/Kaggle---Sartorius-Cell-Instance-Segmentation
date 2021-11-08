import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np  
import pandas as pd 
import cv2
import torch
import re


imageName = '0c90b86742b2'
df= pd.read_csv("train.csv")
row = df[df['id'] == imageName]
print(row)

def convertPixelToCoord(intLocation, width):
    x = intLocation%width
    y = intLocation//width
    return x, y

def RLEncodingToCoords(pixel, run, imgWidth):
    endPixel = pixel + (run-1)
    x1, y1 = convertPixelToCoord(pixel, imgWidth)
    x2, y2 = convertPixelToCoord(endPixel, imgWidth)
    return [(x1, y1), (x2, y2)]

def getRunList(pixel, run, imgWidth):
    list = []
    for i in range(run):
        list.append(convertPixelToCoord(pixel + i, imgWidth))
    return list


annotations = row['annotation']
width = row.iloc[0]['width']
height = row.iloc[0]['height']

image = cv2.imread(f"train/{imageName}.png")
imageWithMask = cv2.imread(f"train/{imageName}.png")

for annotation in annotations:
    color = list(np.random.choice(range(256), size=3))
    listPairsOfMask = re.findall("[0-9]+ [0-9]+", annotation)

    for pair in listPairsOfMask:
        pixelLoc, run = pair.split()
        # -1 car l'indexage des pixels dans le jeu de données a l'air de commencer à 1, contrairement à opencv
        listPixelsCoord = getRunList(int(pixelLoc)-1, int(run), width) 
        for coords in listPixelsCoord:
            if coords[1] < height and coords[0] < width:
                imageWithMask[coords[1], coords[0]] = color
            else: 
                print(f"Incorrect Mask {pixelLoc} {run}")
        


cv2.imshow("With mask", image)
cv2.imshow("With mask", imageWithMask)
cv2.waitKey()

# img = mpimg.imread('train/0a6ecc5fe78a.png')
# imgplot = plt.imshow(img)
# plt.show()