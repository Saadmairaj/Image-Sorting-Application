# -------- -------- -------- -------- IMPORTS -------- -------- -------- -------- 

import os
import cv2
import numpy as np
# from Smart import print
from object_detection_tf.utils.visualization_utils import \
    visualize_boxes_and_labels_on_image_array
from object_detection_tf.utils.extra_tools import object_names,\
    download_model, easy_detect_tf, get_in_order, get_link_by_name

# -------- -------- -------- -------- VARIABLES -------- -------- -------- -------- 

NUM_CLASSES = 90
CAM_INPUT = '/Users/saad/Downloads/MI.mp4'   # path to video sample
PATH_TO_CKPT = download_model(get_link_by_name('SSDLITE Mobilenet V2') , 'COCO-trained_models')  # model to use for detection
PATH_TO_LABELS = os.path.join('object_detection_tf/data', 'mscoco_label_map.pbtxt')   # path to model detection labels.

# -------- -------- -------- -------- INITIALIZE VC -------- -------- -------- -------- 

cap = cv2.VideoCapture(CAM_INPUT)   # initialise input source of video.
reduce_img = 2                      # reduce image size by 2.

# -------- -------- -------- -------- MAINLOOP -------- -------- -------- -------- 

while cv2.waitKey(1) < 0:
    rec, image = cap.read()
    if not rec: break
    image = cv2.resize( image, ( int(image.shape[1] / reduce_img), 
                                 int(image.shape[0] / reduce_img) ))

    # DETECTS
    (boxes, scores, classes, num_detections, category_index) = easy_detect_tf(
            image, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES,
            ['image_tensor:0', 'detection_boxes:0', 'detection_scores:0',
             'detection_classes:0', 'num_detections:0']  )
             
    # DRAW BOXES AND LABELS.
    visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.4)

    # DISPLAYS THE IMAGE
    cv2.imshow('object detection', image)
    # print.fps()     # fps counter.
