import cv2
import numpy as np
import random
import colorsys
import tensorflow as tf
def PreprocessImage(image, imageSize=(320,320)):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, imageSize)
    image = np.array(image, dtype='float32')
    image = np.expand_dims(image, 0)

    return image


def Thresholding(scores, boxes, classes, maxBoxes=10, scoreThreshold=0.5):

    outBoxes = []
    outScores = []
    outClasses = []

    for i in range(min(maxBoxes, boxes.shape[0])):
        if scores is None or scores[i] > scoreThreshold:
            outBoxes.append(boxes[i])
            outScores.append(scores[i])
            outClasses.append(classes[i])

    outBoxes = np.array(outBoxes)
    outScores = np.array(outScores)
    outClasses = np.array(outClasses)

    return outScores, outBoxes, outClasses


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




def Detection(image, sess):
    
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')

    boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image})
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
    outScores, outBoxes, outClasses = Thresholding(scores, boxes, classes, maxBoxes=50, scoreThreshold=0.5)
      
    return outScores, outBoxes, outClasses


def LoadModel(modelPath):
    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        graphDef = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(modelPath, 'rb') as fid:
            serializedGraph = fid.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name='')

    with detectionGraph.as_default():
        sess = tf.Session()
        return sess


def DrawBoxes(image, imageID, objectID, jsonAnnotations, outScores, outBoxes, outClasses, classNames, colors):

    h, w, _ = image.shape

    for i, c in list(enumerate(outClasses)):
        predicted_class = classNames[c]
        box = outBoxes[i]
        score = outScores[i]

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
        
        anno = {
            'segmentation': [[0,0]], 'bbox': [int(left), int(top), int(right-left), int(bottom-top)], 'area': int((right-left)*(bottom-top)),
            #'segmentation': [[0,0]], 'bbox': [left, top, right-left, bottom-top], 'area': (right-left)*(bottom-top),
            'iscrowd': 0, 'image_id': imageID, 'category_id': int(c), 'id': objectID
            }
        
        jsonAnnotations.append(anno)
        objectID += 1

    return image