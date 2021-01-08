import os
import time
import numpy as np

from object_detection_tf.utils.visualization_utils import \
    visualize_boxes_and_labels_on_image_array
from object_detection_tf.utils.extra_tools import object_names,\
    download_model, easy_detect_tf, get_in_order, get_link_by_name

# imports for Testing OtCore for multiprocessing
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt

extentions = ('.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp')

def get_files_path(folder_path, validate, *extentions):
    """Get pathes of all files with specific extentions 
    from all nested folders."""
    tmp_list = []
    for (root, dirs, files) in os.walk(folder_path, topdown=True): 
        for file in files:
            if file.lower().endswith(extentions):
                complete_path = os.path.abspath(root+'/'+file)
                try: 
                    if validate:
                        Image.open(complete_path).verify()
                    tmp_list.append(complete_path)
                except Exception as e:
                    print('Not valid: {}'.format(root+'/'+file))
    return tmp_list


def detect_visualize(image, visualize=False, thresh=0.7, model_name='SSDLITE Mobilenet V2'):
    """Detect and visualize altogether, returns numpy image and name list."""
    
    PATH_TO_CKPT = download_model(get_link_by_name(model_name), 'COCO-trained_models' )
    PATH_TO_LABELS = os.path.join('object_detection_tf/data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    (boxes, scores, classes, num_detections, category_index) = easy_detect_tf(
            image, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES,
            ['image_tensor:0', 'detection_boxes:0', 'detection_scores:0',
             'detection_classes:0', 'num_detections:0']  )

    name_list = object_names(np.squeeze(classes), np.squeeze(scores),
                    np.squeeze(num_detections), category_index)
    
    for i in name_list.copy():
        ob, score = i
        if score < thresh:
            name_list.remove(i)

    # DRAW BOXES AND LABELS.
    if visualize:
        visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=thresh)
    
    return image, name_list


class detection_Thread(QtCore.QThread):

    progressSignal1 = QtCore.pyqtSignal(int)
    progressSignal2 = QtCore.pyqtSignal(int)
    imageSignal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, img_list, select_path, save_path, search, parent=None):
        super().__init__(parent)

        self.search = search
        self.img_list = img_list
        self.save_path = save_path
        self.select_path = select_path

    def run(self):

        incri_by = 0

        try:
            for img in self.img_list:
                start_timne = time.time()
                incri_by += 1
                p_img = Image.open(os.path.join(self.select_path, img))
                np_img, name_list = detect_visualize(np.array(p_img), 
                                        self.showvisualckb.isChecked(), 
                                        self.threshold_value)
                
                for name, score in name_list:
                    if not self.checkBox.isChecked() and self.search is not None:
                        
                        if self.search.lower() == name.lower():
                            self.imageSignal.emit(np_img)
                            if not os.path.exists(self.save_path+'/sorted/'+name):
                                os.makedirs(self.save_path+'/sorted/'+name)
                            if self.savevisualckb.isChecked():
                                p1_img = Image.fromarray(np_img)
                                p1_img.save(self.save_path+'/sorted/'+name+'/'+img)
                            else: 
                                p1_img = Image.fromarray(p_img)
                                p1_img.save(self.save_path+'/sorted/'+name+'/'+img)
                        else:
                            self.imageSignal.emit(np.array(p_img))
                            
                    elif self.checkBox.isChecked():
                        self.imageSignal.emit(np_img)
                        if not os.path.exists(self.save_path+'/sorted/'+name):
                            os.makedirs(self.save_path+'/sorted/'+name)
                        if self.savevisualckb.isChecked():
                            p1_img = Image.fromarray(np_img)
                            p1_img.save(self.save_path+'/sorted/'+name+'/'+img)
                        else: 
                            p1_img = Image.fromarray(p_img)
                            p1_img.save(self.save_path+'/sorted/'+name+'/'+img)
                    
                print(img, incri_by)
                self.progressSignal1.emit(incri_by)
            
                self.est_avg_time.append( time.time() - start_timne )
                if self.est_avg_time.__len__() > 20:
                    self.est_avg_time.pop(0)

                est_time = (sum(self.est_avg_time) / self.est_avg_time.__len__()) \
                                * (self.dial.value() - incri_by)
                self.estTime_label.setText("<html><head/><body><p align=\"center\"><span \
                    style=\" font-size:11pt;\">est: {:.2f} second(s)</span></p></body></html>".
                    format(est_time))

        except Exception as e: 
            print(e)