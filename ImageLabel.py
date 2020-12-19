import os
import sys
import cv2
import json
import glob
import datetime
import argparse
from tqdm import tqdm
from Utils import *


def GetArgs():
    parser = argparse.ArgumentParser(description='Auto Labeling')

    parser.add_argument('--image_path', type=str, metavar='Image Path', help='Type image path. The path should have only images.')
    parser.add_argument('--output_path', type=str, metavar='Output Path', help='Type output path. The path should not have any contents.')
    parser.add_argument('--model_path', type=str, metavar='Model Path', default='./model/mobilenet_ssd_v2.pb', help='Type model path. (inference graph)\n default : mobilenet ssd v2')
    parser.add_argument('-b', '--bbox', action='store_true', help='Save images with bbox')
    
    imagePath = parser.parse_args().image_path
    outputPath = parser.parse_args().output_path
    modelPath = parser.parse_args().model_path
    bboxFlag = parser.parse_args().bbox

    return imagePath, outputPath, modelPath, bboxFlag

if __name__ == '__main__':

    imagePath, outputPath, modelPath, bboxFlag = GetArgs()

    imageFiles = sorted(glob.glob(imagePath + '/*'), key=os.path.getctime)

    if len(imageFiles) < 1:
        print("No image")
        sys.exit()

    os.makedirs(outputPath, exist_ok=True)

    #path = './model/faster_rcnn_resnet50.pb'

    sess = LoadModel(modelPath)
    category = ReadCategory('./model/coco_category.txt')
    colors = GenerateColor(category)

    jsonImages = []
    jsonAnnotations = []
    
    imageID = 1
    objectID = 1
    for imageFile in tqdm(imageFiles):

        imageName = imageFile[len(imagePath)+1:]
        print(imageName)
        image = cv2.imread(imageFile)

        h, w, _ = image.shape

        imageData = PreprocessImage(image)
        outScores, outBoxes, outClasses = Detection(imageData, sess)
        
        cv2.imwrite(outputPath + '/' + imageName, image)

        if bboxFlag == True:
            bboxImage = DrawBoxes(image, imageID, objectID, jsonAnnotations, outScores, outBoxes, outClasses, category, colors)
            os.makedirs(outputPath + '/bbox_images', exist_ok=True)
            cv2.imwrite(outputPath + '/bbox_images/' + imageName, bboxImage)

        jsonImage = {'width': w,' height': h, 'id': imageID, 'file_name': imageName}

        jsonImages.append(jsonImage)
        objectID += len(outClasses)
        imageID += 1

    jsonCategories = MakeJsonCategory(category)
    now = datetime.datetime.now()

    info = {'contributor': 'user', 'description': 'Auto Labeled Dataset', 'version': '1.0', 'date_created': now.strftime('%Y/%m/%d'), 'year': now.strftime('%Y')}
    jsonData = {'annotations': jsonAnnotations, 'images': jsonImages, 'categories': jsonCategories, 'info': info}

    with open(outputPath + '/annotation.json', 'w', encoding='utf-8') as f:
        json.dump(jsonData, f)