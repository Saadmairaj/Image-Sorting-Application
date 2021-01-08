import cv2 as cv
import numpy as np
# from Smart import print

import threading as th
import multiprocessing.pool as mp
import os, time

from object_detection_tf.utils.visualization_utils import \
        visualize_boxes_and_labels_on_image_array
from object_detection_tf.utils.extra_tools import object_names,\
        download_model, get_in_order, _thread_it, get_category_index, \
        tf_dnn_checker, dn_dnn_checker, easy_detect_cv_darknet

OB_DETECTION = 'tensor'
confThreshold = 0.3 if OB_DETECTION.lower() == 'yolo' else 0.2
classesFile = "opencv-trained_models/coco.names"
classes = get_category_index('object_detection_tf/data/mscoco_label_map.pbtxt',
         num_classes=90, use_display_name=True, simple_format=True)


if OB_DETECTION.lower() == 'yolo':
    with open(classesFile,'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    category_index_yolo = {}
    for c in range(len(classes)):
        count = c + 1
        id_name = { 'id': count, 'name': classes[c] }
        category_index_yolo.update({ count: id_name })

    # print(category_index_yolo)

#Model configuration
modelConf, modelWeights = dn_dnn_checker(   
                            'yolov3-spp', 
                            './opencv-trained_models', 
                            download=True)  \
                            if OB_DETECTION.lower() == 'yolo' \
                            else tf_dnn_checker(
                            'ssdlite_mobilenet_v2_coco_2018_05_09', 
                            './opencv-trained_models', True)

def drawPred(classId, conf, x1, y1, x2, y2, thickness=2):
    cv.rectangle(info['frame'], (x1, y1), (x2, y2), (255, 178, 50), thickness)
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    cv.rectangle(info['frame'], (x1-thickness, y1), (x2+thickness, y1-17), (255, 178, 50), cv.FILLED, cv.LINE_AA)
    cv.putText(info['frame'], label, (x1, y1-6), cv.FONT_HERSHEY_COMPLEX, 0.4, (255,255,255), 1, cv.LINE_AA)

net = cv.dnn.readNet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#Process inputs
winName = 'DL OD with OpenCV'
cap = cv.VideoCapture('/Users/saad/Downloads/MI.mp4')
cap.set(cv.CAP_PROP_FPS, 25)

hasFrame, frame = cap.read()

div = 1.0
num = 416 if OB_DETECTION.lower() == 'yolo' else 300
inpHeight = int( num / div)
inpWidth = int( num/ div )

info = {}
def _thread_it(normal=False):
    while 1:
        if info.get('blob', None) is not None:
            layerNames = [  net.getLayerNames()[ i[0] - 1 ] 
                    for i in net.getUnconnectedOutLayers()  ]
            net.setInput(info['blob'])
            info['outs'] = net.forward(layerNames)
            if normal: return info['outs']
        # print.fps('Thread')
# mp1 = mp.ThreadPool(processes=1)
# async_result = mp1.apply_async(_thread_it)

category_index = get_category_index('object_detection_tf/data/mscoco_label_map.pbtxt',
                            num_classes=90, use_display_name=True)

while cv.waitKey(1) < 0:
    rec, frame = cap.read()
    if not rec: break    
    h,w,ch = frame.shape
    # frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    info['frame'] = cv.resize(frame, (int(w/1.5), int(h/1.5)))
    
    
    if OB_DETECTION.lower() == 'yolo':
        boxes, scores, classes_, num_detection, category_index_yolo = easy_detect_cv_darknet(
            info['frame'], modelConf, modelWeights, labelFile='./opencv-trained_models/coco.names', 
            size=(inpWidth, inpHeight), scale=0.5, min_score_thresh=0.25, threadIt=1  )
            
        visualize_boxes_and_labels_on_image_array(
                info['frame'],
                np.squeeze(boxes),
                np.squeeze(classes_).astype(np.int32),
                np.squeeze(scores),
                category_index_yolo,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.3)

    else:
        # print(1.0/127.5)
        info['blob'] = cv.dnn.blobFromImage(
                image = cv.resize(info['frame'], (300,300)), 
                # scalefactor = 0.3,
                size = (300, 300), 
                mean = (127.5, 127.5, 127.5),
                swapRB = True, 
                crop = False,
                ddepth=cv.CV_8U)
                
        rows, cols, ch = info['frame'].shape
        boxes, classes_, scores = [], [], []
        tmp = np.squeeze(info.get('outs', []))
        info['outs'] = _thread_it(1)
        for i in range(100):
            if tmp.size != 0 and len(tmp) >= i and tmp[i][2] > 0.15:
                detection = tmp[i]
                #                    y1             x1           y2             x2
                bbox = np.array([detection[4], detection[3], detection[6], detection[5]])
                classes_.append(np.array(detection[1]+1.0))
                boxes.append(bbox)
                scores.append(detection[2])   
            else:
                bbox = np.array([1.0, 1.0, 1.0, 1.0])
                classes_.append(np.array(1.0))
                boxes.append(bbox)
                scores.append(0.0)

        visualize_boxes_and_labels_on_image_array(
                info['frame'],
                np.squeeze(boxes),
                np.squeeze(classes_).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.35)

    cv.imshow(winName, info['frame'])
    # cv.waitKey(0)
    # print.fps('Read and Display')
