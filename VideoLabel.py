from argparse import Action
from ast import parse
import os
import sys
import cv2
import json
import glob

import numpy as np
import tensorflow as tf
import datetime
import argparse
from tqdm import tqdm
from Utils import *


def GetArgs():

    parser = argparse.ArgumentParser(description='Auto Labeling')

    parser.add_argument('--video_path', type=str, metavar='Video Path', help='Type video path. The path should have only videos.')
    parser.add_argument('--output_path', type=str, metavar='Output Path', help='Type output path. The path should not have any contents.')
    parser.add_argument('-b', '--bbox', action='store_true', help='Save images with bbox')
    

    videoPath = parser.parse_args().video_path
    outputPath = parser.parse_args().output_path
    bboxFlag = parser.parse_args().bbox


    return videoPath, outputPath, bboxFlag


if __name__ == '__main__':

    videoPath, outputPath, bboxFlag = GetArgs()

    videoFiles = glob.glob(videoPath + '/*')

    if len(videoFiles) < 1:
        print("No video")
        sys.exit()

    os.makedirs(outputPath, exist_ok=True)

    path = './model/faster_rcnn_resnet50.pb'

    sess = LoadModel(path)

    category = ReadCategory('./model/coco_category.txt')
    colors = GenerateColor(category)

    jsonImages = []
    jsonAnnotations = []
    
    imageID = 1
    objectID = 1
    for videoFile in tqdm(videoFiles):
        cap = cv2.VideoCapture(videoFile)
        frameCount = 0

        while cap.isOpened():
            _, frame = cap.read()

            if not _ :
                break

            frameCount += 1
            if frameCount % 20 == 0:
                h, w, _ = frame.shape
                frameData = PreprocessImage(frame, (320,320))
                outScores, outBoxes, outClasses = Detection(frameData, sess)

                bboxImage = DrawBoxes(frame, imageID, objectID, jsonAnnotations, outScores, outBoxes, outClasses, category, colors)
                
                cv2.imwrite(outputPath + '/' + str(frameCount) + '.jpg', frame)

                if bboxFlag == True:
                    os.makedirs(outputPath + '/bbox_images', exist_ok=True)
                    cv2.imwrite(outputPath + '/bbox_images/' + str(frameCount) + '.jpg', bboxImage)

                jsonImage = {'width': w,' height': h, 'id': imageID, 'file_name': str(frameCount) + '.jpg'}

                #jsonAnnotations.append(jsonAnnotation)
                jsonImages.append(jsonImage)
                objectID += len(outClasses)
                imageID += 1

    now = datetime.datetime.now()
    info = {'contributor': 'user', 'description': 'Auto Labeled Dataset', 'version': '1.0', 'date_created': now.strftime('%Y/%m/%d'), 'year': now.strftime('%Y')}
    jsonCategories = MakeJsonCategory(category)
    
    jsonData = {'annotations': jsonAnnotations, 'images': jsonImages, 'categories': jsonCategories, 'info': info}

    with open(outputPath + '/annotation.json', 'w', encoding='utf-8') as f:
        json.dump(jsonData, f)