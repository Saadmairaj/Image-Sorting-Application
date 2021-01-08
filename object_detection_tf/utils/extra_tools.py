
import urllib.request as ur
import tarfile
import os
import cv2
import time
import tensorflow as tf
import numpy as np
import threading as th

from multiprocessing.pool import ThreadPool

from object_detection_tf.utils.label_map_util import load_labelmap, \
    convert_label_map_to_categories, create_category_index


class modified_dictionary(dict):
    def get(self, key, d=None):
        """Not case sensitive."""
        for k, v in self.items():
            if key.lower() == k.lower(): return v
        return d 
    def get_by_word(self, word, d=None):
        """Get all values whose keys contains the given word. 
        Not case sensitive."""
        tmp = []
        for k, v in self.items():
            if word.lower() in k.lower(): tmp.append(v)
        if tmp: return tmp
        return d


SSD_Model_Names = modified_dictionary({
    'SSD Mobilenet V1'  :   
        ('ssd_mobilenet_v1_coco_2018_01_28.tar.gz', (30, 21)),
    'SSD Mobilenet V1 0.75 depth 300x300'  :   
        ('ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03.tar.gz', (26, 18)),
    'SSD Mobilenet V1 Quantized 300x300'  :   
        ('ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz', (29, 18)),
    'SSD Mobilenet V1 0.75 depth Quantized 300x300'  :   
        ('ssd_mobilenet_v1_0.75_depth_quantized_300x300_coco14_sync_2018_07_18.tar.gz', (29, 16)),
    'SSD Mobilenet V1 PPN shared box prediction 300x300'  :   
        ('ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03.tar.gz', (26, 20)),
    'SSD Mobilenet V1 FPN shared box prediction 640x640'  :   
        ('ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz', (56, 32)),
    'SSD Resnet50 V1 FPN shared box prediction 640x640'  :   
        ('ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz', (76, 35)),
    'SSD Mobilenet V2'  :   
        ('ssd_mobilenet_v2_coco_2018_03_29.tar.gz', (31, 22)),
    'SSD Mobilenet V2 Quantized 300x300'  :   
        ('ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz', (29, 22)),
    'SSDLITE Mobilenet V2'  :   
        ('ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz', (27, 22)),
    'SSD Inception V2'  :   
        ('ssd_inception_v2_coco_2018_01_28.tar.gz', (42, 24)),
})


RCNN_Model_Names = modified_dictionary({
    'Faster RCNN Inception V2'  :   
        ('faster_rcnn_inception_v2_coco_2018_01_28.tar.gz', (58, 28)),
    'Faster RCNN Resnet50'  :   
        ('faster_rcnn_resnet50_coco_2018_01_28.tar.gz', (89, 30)),
    'Faster RCNN Resnet50 Lowproposals'  :   
        ('faster_rcnn_resnet50_lowproposals_coco_2018_01_28.tar.gz', (64, '--')),
    'RFCN Resnet101'  :   
        ('rfcn_resnet101_coco_2018_01_28.tar.gz', (92, 30)),
    'Faster RCNN Resnet101'  :   
        ('faster_rcnn_resnet101_coco_2018_01_28.tar.gz', (106, 32)),
    'Faster RCNN Resnet101 Lowproposals'  :   
        ('faster_rcnn_resnet101_lowproposals_coco_2018_01_28.tar.gz', (82, '--')),
    'Faster RCNN Inception Resnet Atrous V2'  :   
        ('faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz', (620, 37)),
    'Faster RCNN Inception Resnet Atrous Lowproposals V2'  :   
        ('faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28.tar.gz', (241, '--')),
    'Faster RCNN Nas'  :   
        ('faster_rcnn_nas_coco_2018_01_28.tar.gz', (1833, 43)),
    'Faster RCNN Nas Lowproposals'  :   
        ('faster_rcnn_nas_lowproposals_coco_2018_01_28.tar.gz', (540, '--')),
})


MASK_Model_Names = modified_dictionary({
    'Mask RCNN Inception Resnet V2 Atrous'  :   
        ('mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz', (771, 36)),
    'Mask RCNN Inception'  :   
        ('mask_rcnn_inception_v2_coco_2018_01_28.tar.gz', (79, 25)),
    'Mask RCNN Resnet101 Atrous'  :   
        ('mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz', (470, 33)),
    'Mask RCNN Resnet50 Atrous'  :   
        ('mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz', (343, 29)),
})



def get_link_by_name(name, multiple=False):
    """Give model download link by their name like ( 'SSD V2 Mobilenet' ) 
    and this will return the relevant download links."""
    tmp_list=[]
    download_link = 'http://download.tensorflow.org/models/object_detection/'
    for i in (SSD_Model_Names, RCNN_Model_Names, MASK_Model_Names):
        for model_name, info in i.get_by_word(name, []):
            tmp_list.append(download_link+model_name)
    if tmp_list and multiple: return tmp_list
    elif tmp_list: return tmp_list[0]
    return None


def object_names(classes, scores, num_detections, category_index):
    """Gives list of names of objects contained in the image."""
    names_list = []
    for i in range(int(num_detections)):
        if classes[i] in category_index.keys():
            names_list.append((category_index[classes[i]]['name'], scores[i]))
    return names_list


def get_in_order(boxes, scores, classes, num_detections, category_index):
    """
    ### Get In Order
    Arranges all the information given and returns a list of dictionaries.

    #### Args:
    - `boxes`: Detected boxes.
    - `scores`: Detection scores.
    - `classes`: Classification list.
    - `num_detections`: Number of detections.
    - `category_index`: Category list with id and name.
    """
    orderedlist = []
    info_list = {
        'boxes': boxes, 'scores': scores, 'classes': classes,
        }    
    for k, v in info_list.copy().items():
        if isinstance(v, np.ndarray):
            info_list[k] = np.squeeze(v).tolist()

    for i in range(int(num_detections)):
        tmp = {}
        if info_list['classes'][i] in category_index.keys():
            tmp['id'] = category_index[info_list['classes'][i]]['id']
            tmp['name'] = category_index[info_list['classes'][i]]['name']
        else:
            tmp['name'] = 'N/A'
            tmp['id'] = 'N/A'
        try:
            tmp['bbox'] = info_list['boxes'][i]
            tmp['score'] = info_list['scores'][i]
        except Exception: pass
        orderedlist.append(tmp)
    return orderedlist


def unziper(path, save_directory, find='all'):
    """
    ### Unziper 
    Unzip and extract any .tar file.

    #### Args:
    - `find`: Find and extract a specfic file.
    - `path`: Path to the zip file.
    - `save_directory`: Path to save the extracted file.
    """
    if not isinstance(find, (list, tuple)) and find != 'all':
        find = tuple(find)
    if not save_directory.endswith('/'):
        save_directory += '/'

    tar_file = tarfile.open(path)
    file = tar_file.getmembers()
    if find == 'all':
        tar_file.extractall(path=save_directory, members=file)
    else:
        for f in file:
            file_name = os.path.basename(save_directory+'/'+f.name)
            for fin in find:
                if fin.startswith('.') and file_name.endswith(fin):
                    tar_file.extract(f, save_directory)
                elif fin in file_name:
                    tar_file.extract(f, save_directory)
        

def download_model(model_name, save_directory='trained_models', 
                    download_base='http://download.tensorflow.org/models/object_detection/',
                    unzip=True, unzip_find='.pb', overwrite=False):
    """
    ### Download Pre-trained Models from this function.

    #### Args: 
    - `model_name`: Give the name of the model.
    - `save_directory`: Path to save the model.
    - `download_base`: URL to download the model from.
    - `unzip`: Set it True to unzip the file as well.
    - `unzip_find`: Give a specfic to be unziped.
    - `overwrite`: Downloads and re-writes the file if exists.

    #### Returns:
    Path of "frozen_inference_graph.pb" file.
    """
    exentention = ' '
    download_link = model_name
    if download_base not in model_name:
        download_link = download_base + model_name
    else:
        model_name = model_name.split('/')[-1]
    if model_name.endswith('.tar.gz'):
        model_name = model_name[:-7]
        exentention = '.tar.gz'
    elif model_name.endswith('.tar'):
        model_name = model_name[:-4]
        exentention = '.tar.gz'
    elif model_name.endswith('.weights'):
        model_name = os.path.splitext(model_name)[0]
        unzip = False
        exentention = '.weights'

    model_file = model_name + exentention
        
    rv = save_directory+'/'+model_name
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    if os.path.exists(save_directory+'/'+model_file) and not overwrite:
        if unzip:
            unziper(save_directory+'/'+model_file, save_directory, 
                    find=unzip_find )
            for i in os.listdir(rv):
                if i.endswith(unzip_find):
                    rv += '/'+i
            return rv
        elif exentention == '.weights': return rv+exentention

    
    opened_url = ur.urlopen(download_link)
    file_size = int(dict(opened_url.info().items())['Content-Length'])    
    print ("\nDownloading: %s\nFrom: %s\nSize: %.3f MB\nUnzip: %s\nOverwrite: %s" % (
        model_file, download_base, file_size * 1e-6, unzip, overwrite))
    file = open(save_directory+'/'+model_file, 'wb')
    
    downloaded = 0
    block_sz = int(8192 * 9)
    while True:
        buffer = opened_url.read(block_sz)
        if not buffer: break
        downloaded += len(buffer)
        file.write(buffer)
        status = "\t | [ %3.2f%% ] | [ %.3f MB of %.3f MB ] |" % (
                downloaded * 100. / file_size, downloaded * 1e-6, file_size * 1e-6)
        status = status + chr(8)*(len(status)+1)
        print(status, end='\r')
    print()
    file.close()

    if unzip:
        unziper(save_directory+'/'+model_file, save_directory, 
                find=unzip_find)
        for i in os.listdir(rv):
            if i.endswith(unzip_find):
                rv += '/'+i
        return rv
    elif exentention == 'weights': 
        return rv+exentention


def get_category_index(path, num_classes, use_display_name=True, simple_format=False):
    """
    ### Category Index.
    Fetches and creates category_index dictionary from .pbtxt file.

    #### Args:
    - `path`: Path to the .pbtxt file.
    - `num_classes`: Number of classes required.
    - `use_display_name`: Display names or code names.
    - `simple_format`: Returns a simpler version with just names.

    #### Returns:
    Category_index dictionary
    """
    label_map = load_labelmap(path)
    categories = convert_label_map_to_categories(label_map, 
                max_num_classes=num_classes, 
                use_display_name=use_display_name)
    rv = create_category_index(categories)
    if simple_format:
        for k,v in rv.copy().items():
            rv[k] = v['name']
    return rv


def tf_dnn_checker(model_name, path, download=False):
    """
    ### Tensorflow dnn files checker.
    This will check if any file exists or not. 
    If any file does not exists this function will fetch 
    and ready the required files for `cv2.dnn.readNetFromTensorflow()`. 

    #### Args:
    - `model_name`: Complete name of the tensorflow COCO-trained model.
    - `path`: Path to directory to Save or to open files.
    - `download`: Download the model if doesn't exists.

    #### Returns:
    Paths to .pb and .pbtxt files.
    """

    if not os.path.exists(path+'/'+model_name):
        if download:
            download_model(model_name+'.tar.gz', path, unzip_find=('.pb', '.config'))
        else: assert ('File not found!')
            
    path_to_pb = ''
    path_to_pbtxt = ''
    path_to_config = ''
    script_path = 'object_detection_tf/ulits_cv/'
    path_to_model = os.path.abspath(os.path.join(path, model_name))
    for i in os.listdir(path_to_model):
        if i.endswith('.config'):
            path_to_config = os.path.abspath(os.path.join(path_to_model, i))
        elif i.endswith('.pb'):
            path_to_pb = os.path.abspath(os.path.join(path_to_model, i))
        elif i.endswith('pbtxt'):
            path_to_pbtxt = os.path.abspath(os.path.join(path_to_model, i))
    
    if not path_to_pbtxt:
        cv2.dnn.writeTextGraph(path_to_pb, model_name+'/graph.pbtxt')

    for i in os.listdir(path_to_model):
        if i.endswith('pbtxt'):
            path_to_pbtxt = os.path.abspath(os.path.join(
                                path_to_model, i))
    return path_to_pb, path_to_pbtxt


def dn_dnn_checker(model_name, path, download=False):
    """
    ### Darknet dnn files checker.
    This will check if any file exists or not. 
    If any file does not exist this function will fetch 
    and ready the required files for `cv2.dnn.readNetFromDarknet()`. 

    #### Args:
    - `model_name`: Complete name of the Darknet COCO-trained model (YOLO).
    - `path`: Path to directory to Save or to open files.
    - `download`: Download the model if doesn't exists.

    #### Returns:
    Paths to .cfg and .weight files.
    """
    path = os.path.join(path, model_name)
    path_to_cfg = os.path.join(path, model_name+'.cfg')
    path_to_weight = os.path.join(path, model_name+'.weights')
    site_to_get_cfg = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/'
    if download:
        download_model(model_name+'.weights', path, 
                download_base='https://pjreddie.com/media/files/')
    else: assert ('File not found!')
    if not os.path.exists(path_to_cfg):
        with open(path_to_cfg, 'wb') as file:
            cfg = ur.urlopen(site_to_get_cfg+model_name+'.cfg')
            file.write(cfg.read())
    return path_to_cfg, path_to_weight
    

def load_ckpt_to_graph(path):
    """
    ### Loads ckpt data to the graph and returns the graph

    #### Args:
    - `path`: Path to ckpt data file.
    """
    tf.reset_default_graph()
    # tf.InteractiveSession.close()
    detection_graph = tf.get_default_graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def _create_session(graph=tf.get_default_graph(), threads=44):
    """Wrapper function for `easy_detect`. 
    
    It returns an existing `tf.Session()` if exists but 
    if not then creates a new one"""
    sess = tf.get_default_session()
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = threads
    config.inter_op_parallelism_threads = threads
    # tf.Session(config=config)
    if sess is None:
        sess = tf.Session(graph=graph, config=config)
        sess.__enter__()
    return sess


def _thread_it(thread_name='TF.thread', func=None):

    # thread_names = [ i.name for i in th.enumerate() ]
    # if thread_name in thread_names: return

    thr = th.Thread(name=thread_name, target=func)
    thr.start()
    return thr


def easy_detect_tf(image, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES=90,
                tensor_name_list=[] ):
    """### Detect objects easily.
    
    #### Args:
    - `image`: Give numpy image.
    - `PATH_TO_CKPT`: Give path to .pb file of the model.
    - `PATH_TO_LABELS`: Give path to .pbtxt data for labels.
    - `NUM_CLASSES`: Number of classes to do detection for.
    - `tensor_name_list`: Give the list of tensor names. (`'example:0'`)
    
    #### Returns:
    - returns tuple of all of numpy given to `tensor_name_list`. 
        Category_index list will be at last index of returned tuple."""

    tensor_list = []
    image_expand = np.expand_dims(image, axis=0)
    image_tensor = None
    detection_graph = tf.get_default_graph()
    try:
        detection_graph.get_tensor_by_name(tensor_name_list[0])
    except: 
        detection_graph = load_ckpt_to_graph(PATH_TO_CKPT)
    sess = _create_session(graph=detection_graph)

    for i in tensor_name_list:
        if 'image_tensor:0' in i:
            image_tensor = detection_graph.get_tensor_by_name(i)
        else:
            tensor_list.append(
                detection_graph.get_tensor_by_name(i) )

    category_index = detection_graph.get_collection('category_index')
    if category_index == []:
        label_map = load_labelmap(PATH_TO_LABELS)
        categories = convert_label_map_to_categories(label_map, 
                    max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = create_category_index(categories)
        detection_graph.add_to_collection('category_index', category_index)
    else: category_index = category_index[0]
    
    rv = []
    def run_sess():
        rv.extend(sess.run(tensor_list, 
                feed_dict={image_tensor: image_expand} ) )
        return rv

    # _thread_it(func=run_sess)
    run_sess()
    rv.append(category_index)
    return rv


_info = {}
def easy_detect_cv_darknet(image, config, weight, labelFile='Auto', 
                           scalefactor=1/255, size=(320, 320), scale=0.5, 
                           min_score_thresh=0.2, threadIt=False):
    """
    ### Detect objects easily.
    Object detection with opencv from darknet dataset. (YOLO)

    #### Args:
    - `image`: Numpy array Image.
    - `config`: Path to .cfg file.
    - `weight`: Path to .weight file.
        - NOTE: Get paths of both .cfg and .weight files very easily 
        from `dn_dnn_checker()` (requires internt).
    - `labelFile`: Path to file document containing names of classes. 
        - NOTE: `labelFile='Auto'` will download the file automatically 
        (requires internt).
    - `scalefactor`: Multiplier for image values. 
    - `size`: Spatial size for output image.
    - `min_score_thresh`: Minimum score threshold for a detection.
    - `thread_it`: Uses threading to evaluate detect for faster results.
    
    #### Return:
    List of numpy array ie: (boxes, scores, classes, num_detection, category_index)
    """
    category_index = _info.get('category_index_dk', {})
    net, old_config, old_weight = _info.get('readNet_dk', [None, None, None])
    rows, cols, ch = image.shape
    classes, scores, boxes = [], [], []

    while labelFile == 'Auto' :
        if os.path.exists('coco.names'):
            labelFile = 'coco.names' 
            break
        url = ur.urlopen(
                'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
        with open('coco.names', 'wb') as file: file.write(url.read())
        labelFile = 'coco.names'
        break

    if category_index == {}:
        with open(labelFile,'rt') as f:
            classes_names = f.read().rstrip('\n').split('\n')
            for c in range(len(classes_names)):
                id_name = { 'id': c + 1, 'name': classes_names[c] }
                category_index.update({ c + 1: id_name })
        _info['category_index_dk'] = category_index
    
    if net is None or old_config != config or old_weight != weight:
        net = cv2.dnn.readNet(weight, config)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        _info['readNet_dk'] = [net, config, weight]
    
    def thread_process(normal=False):
        while image.size != 0:
            layerNames = [  net.getLayerNames()[ i[0] - 1 ] 
                    for i in net.getUnconnectedOutLayers()  ]
            net.setInput(_info['blob_dk'])
            _info['outs_dk'] = net.forward(layerNames)
            if normal: return _info['outs_dk']
        return []

    _info['blob_dk'] = cv2.dnn.blobFromImage(
                image = image, scalefactor = scalefactor,
                size = size, crop = False, mean = [ 0, 0, 0 ], 
                swapRB = True  )
    
    _info['outs_dk'] = _info.get('outs_dk', [])
    if threadIt and not _info.get('thread_dk'):
        MPthread = ThreadPool(processes=1)
        MPthread.apply_async(thread_process)
        _info['thread_dk'] = True
    elif not _info.get('thread_dk'):
        _info['outs_dk'] = thread_process(1)

    for out in range(len(_info['outs_dk'])):
        for detection in _info['outs_dk'][out]:
            score       = detection [5:]
            classID     = np.argmax(score)
            confidence  = score[classID]
            if confidence > min_score_thresh:
                x1 = detection[0] - (detection[2] * scale)
                y1 = detection[1] - (detection[3] * scale)
                x2 = detection[0] + (detection[2] * scale)
                y2 = detection[1] + (detection[3] * scale)
                classes.append(np.array(float(classID+1)))
                scores.append(np.array(confidence))
                boxes.append(np.array([ y1, x1, y2, x2 ]))
    
    for i in range(100):
        if len(boxes) < 100:
            boxes.append(np.array([1.0, 1.0, 1.0, 1.0]))
        if len(classes) < 100:
            classes.append(np.array(1.0))
        if len(scores) < 100:
            scores.append(np.array(0.0))  
    
    return boxes, scores, classes, [len(boxes),], category_index


if __name__ == "__main__":
    s = time.time()
    print(download_model('ssdlite_mobilenet_v2_coco_2018_05_09', 'COCO-trained_models'))
    print(time.time()-s)
