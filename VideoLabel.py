import os
import sys
import cv2
import json
import glob
import random
import colorsys
import time
import numpy as np
import tensorflow as tf
import datetime
import argparse
from tqdm import tqdm
from Utils import *


def GetArgs():
    parser = argparse.ArgumentParser(description='Auto Labeling')

    parser.add_argument('--video_path', '-video_path', type=str, metavar='Video Path', help='Type video path. The path should have only videos.')
    parser.add_argument('--output_path', '-output_path', type=str, metavar='Output Path', help='Type output path. The path should not have any contents.')
    
    #parser.add_argument('')

    videoPath = parser.parse_args().video_path
    outputPath = parser.parse_args().output_path

    return videoPath, outputPath


def Detection(image, sess):
    
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')

    boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image})
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
    out_scores, out_boxes, out_classes = NonMaxSuppression(scores, boxes, classes)
      
    return out_scores, out_boxes, out_classes


def LoadModel(modelPath):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(modelPath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        sess = tf.Session()
        return sess


def ReadCategory(categoryPath):
    with open(categoryPath) as f:
        categories = f.readlines()
    categories = [c.strip() for c in categories]
    return categories

def GenerateColor(categories):
    hsv_tuples = [(x / len(categories), 1., 1.) for x in range(len(categories))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)

    return colors

def MakeJsonCategory(categories):
    jsonCategory = []
    id = 1
    for category in categories:
        if id == 1:
            superClass = 'person'
        elif id > 1 and id < 10:
            superClass = 'outdoor'
        elif id > 9 and id < 16:
            superClass = 'outdoor'
        elif id > 15 and id < 26:
            superClass = 'animal'
        elif id > 25 and id < 34:
            superClass = 'accessory'
        elif id > 33 and id < 44:
            superClass = 'sports'
        elif id > 43 and id < 52:
            superClass = 'kitchen'
        elif id > 51 and id < 62:
            superClass = 'food'
        elif id > 61 and id < 72:
            superClass = 'furniture'
        elif id > 71 and id < 78:
            superClass = 'electronic'
        elif id > 77 and id < 84:
            superClass = 'appliance'
        else:
            superClass = 'indoor'

        jsonC = {'id': id, 'name': category, 'superclass': superClass}
        jsonCategory.append(jsonC)
        id += 1

    return jsonCategory


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    h, w, _ = image.shape

    for i, c in list(enumerate(out_classes)):
    #for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = (xmin * w, xmax * w, ymin * h, ymax * h)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
        right = min(w, np.floor(right + 0.5).astype('int32'))
        cv2.rectangle(image, (left, top), (right, bottom), tuple(colors[c]), 6)

        label = '{} {:.2f}'.format(predicted_class, score)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        label_rect_left, label_rect_top = int(left - 3), int(top - 3)
        label_rect_right, label_rect_bottom = int(left + 3 + label_size[0]), int(top - 5 - label_size[1])
        cv2.rectangle(image, (label_rect_left, label_rect_top), (label_rect_right, label_rect_bottom), tuple(colors[c]), -1)
        cv2.putText(image, label, (left, int(top - 4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
    return image


if __name__ == '__main__':

    videoPath, outputPath = GetArgs()
    
    print(videoPath)
    print(outputPath)

    videoFiles = glob.glob(videoPath + '/*')
    if len(videoFiles) < 1:
        print("No video")
        sys.exit()

    os.makedirs(outputPath, exist_ok=True)

    path = './model/mobilenetv2.pb'

    sess = LoadModel(path)

    category = ReadCategory('./model/coco_category.txt')
    colors = GenerateColor(category)
    jsonImages = []
    jsonAnnotations = []
    
    for videoFile in tqdm(videoFiles):
        cap = cv2.VideoCapture(videoFile)
        frameCount = 0
        while cap.isOpened():
            _, frame = cap.read()
            if not _ :
                break

            frameCount += 1
            if frameCount % 20 == 0:
                frameData = PreprocessImage(frame, (320,320))
                out_scores, out_boxes, out_classes = Detection(frameData, sess)

                bb = draw_boxes(frame, out_scores, out_boxes, out_classes, category, colors)

                cv2.imwrite(outputPath + '/' + str(frameCount) + '.jpg', bb)
                jsonAnnotations.append(frameCount)
                jsonImages.append(frameCount)


    info = {'contributor': 'f'}
    jsonCategories = MakeJsonCategory(category)

    jsonData = {'annotations': jsonAnnotations, 'images': jsonImages, 'categories': jsonCategories}

    with open(outputPath + '/annotation.json', 'w', encoding='utf-8') as f:
        json.dump(jsonData, f)

                