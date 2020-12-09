# AutoLabeling
auto labeling using tensorflow

this is for making custom dataset with own video or images
- - -

### dependencies
- tensorflow 1.15.0
- opencv-python
- tqdm


### how to use
in anaconda terminal, excute VideoLabel.py or ImageLabel.py python file with some commands.

example)

python VideoLabel.py --video_path [dir] --output_path [dir] --model_path [dir] --bbox

python ImageLabel.py --video_path [dir] --output_path [dir] --model_path [dir] --bbox
  
### dataset format
this code generates only coco style annotation file.

and you can't use this code for generate segmentation info.

only Bbox available.



### model
i tested on mobilenet ssd v2 and faster rcnn resnet50. you can download pre-trained models at tensorflow 1 model zoo[https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md]
