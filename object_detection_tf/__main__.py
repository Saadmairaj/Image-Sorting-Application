# YOLO: testing functions to create utils for yolo model. 

# Keeping the paramters same for both tensorflow and yolo 
#   which makes easy to use any model with same parameter 
#   without doing any changes to the detection code.

# -------- -------- -------- -------- -------- -------- -------- --------
import numpy as np
import os
import tensorflow as tf

import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

import sys
sys.path.append(os.getcwd())

from object_detection_tf.utils import label_map_util
from object_detection_tf.utils import visualization_utils as vis_util
from object_detection_tf.utils.extra_tools import object_names, download_model


# List of the strings that is used to add correct label for each box.
PATH_TO_CKPT = 'object_detection_tf/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('object_detection_tf/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1000

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # image_np = cv2.flip(image_np, 1)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)     # >>> IMP
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            print(object_names(np.squeeze(classes).astype(np.int32), np.squeeze(num_detections), category_index))

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2,
                    )

            cv2.imshow('object detection', image_np)
            cv2.waitKey(25)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            