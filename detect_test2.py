# Detection test: Testing object detection with different models 
# using multiple multiprocessing and threading.


# -------- -------- -------- -------- IMPORTS -------- -------- -------- -------- 

# from Smart import print
import os
import numpy as np
import cv2, time
from object_detection_tf.utils.visualization_utils import \
        visualize_boxes_and_labels_on_image_array
from object_detection_tf.utils.extra_tools import object_names,\
        download_model, easy_detect_tf, get_in_order

import multiprocessing.pool as mp
from multiprocessing import set_start_method
import queue

# -------- -------- -------- -------- VARIABLES -------- -------- -------- -------- 

NUM_CLASSES = 1000
CAM_INPUT = '/Users/saad/Downloads/MI.mp4'
PATH_TO_CKPT = download_model('ssdlite_mobilenet_v2_coco_2018_05_09.tar', 'COCO-trained_models')
PATH_TO_LABELS = os.path.join('object_detection_tf/data', 'mscoco_label_map.pbtxt')

# -------- -------- -------- -------- INITIALIZE VC -------- -------- -------- -------- 

cap = cv2.VideoCapture(CAM_INPUT)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, 
            (frame_width,frame_height))

t1 = time.time()
count = 0

def process_draw(image):
    global PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, count
    (boxes, scores, classes, num_detections, category_index) = easy_detect_tf(
            image, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES,
            ['image_tensor:0', 'detection_boxes:0', 'detection_scores:0',
            'detection_classes:0', 'num_detections:0']  )

    # DRAW BOXES AND LABELS.
    rv =  visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.4)
    
    count += 1
    print('Processed frames: ',count)
    return rv

def process(image):
    global PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, count
    return easy_detect_tf(
            image, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES,
            ['image_tensor:0', 'detection_boxes:0', 'detection_scores:0',
            'detection_classes:0', 'num_detections:0']  )

# -------- -------- -------- -------- MAINLOOP -------- -------- -------- -------- 
if __name__ == "__main__":
    set_start_method('spawn', force=True)

    frames = []
    for i in range(1000):
        rec, image = cap.read()
        if not rec: break
        frames.append(image)

    pool = mp.Pool(processes=3)
    map_pool = []
    for i in range(10):
        print(100*i, 100*i+100)
        map_pool.append(
            pool.map_async(process_draw, frames[ 100*i: 100*i+100 ])
        )
    # frames_pool0 = pool.map(process_draw, frames[90:])
    map_pool[0].wait(200)
    frames_pool = []
    for i in map_pool:
        frames_pool += i.get()

    for i in range(len(frames_pool)):
    # for i in frames_pool:
        out.write(frames_pool[i])
    
    print(time.time()-t1)
    out.release()


