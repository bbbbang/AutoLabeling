import os
import colorsys
import random
import cv2

import numpy as np
import tensorflow as tf


def PreprocessImage(image, imageSize=(320,320)):

    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image, 0)

    return image


def NonMaxSuppression(scores, boxes, classes, maxBoxes=10, scoreThreshold=0.5):

    outBoxes = []
    outScores = []
    outClasses = []

    if not maxBoxes:
        maxBoxes = boxes.shape[0]
    for i in range(min(maxBoxes, boxes.shape[0])):
        if scores is None or scores[i] > scoreThreshold:
            outBoxes.append(boxes[i])
            outScores.append(scores[i])
            outClasses.append(classes[i])

    outBoxes = np.array(outBoxes)
    outScores = np.array(outScores)
    outClasses = np.array(outClasses)

    return outScores, outBoxes, outClasses