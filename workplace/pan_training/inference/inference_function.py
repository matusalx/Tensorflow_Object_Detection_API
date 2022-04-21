import os, sys
from os.path import dirname
project_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import pytesseract
import time
#from object_detection.utils import label_map_util

def load_model():

    PATH_TO_SAVED_MODEL = project_dir + r'/pan_training/exported_models/my_faster_rcnn_1024_first/saved_model'
    PATH_TO_LABELS = project_dir + r'/pan_training/annotations/pan_label_map.pbtxt'

    print('Loading model...', end='')
    start_time = time.time()
    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn
detect_fn = load_model()




# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
#                                                                     use_display_name=True)

import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


def predict(image_to_predict):

    image_np = np.array(image_to_predict)
    print('Running inference for {}... ')
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    print('input_tensor = input_tensor[tf.newaxis, ...]')
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = np.expand_dims(image_np, 0)
    print('input_tensor = input_tensor[tf.newaxis, ...]')
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    # detection_classes should be ints.
    print('image_np_with_detections = image_np.copy()')
    image_np_with_detections = image_np.copy()
    #
    first_9_classes = 9

    detections.pop('num_detections')
    detections_top_9 = {key: value[0, :first_9_classes].numpy()
                        for key, value in detections.items()}
    detections_top_9['num_detections'] = first_9_classes
    detections_top_9['detection_classes'] = detections_top_9['detection_classes'].astype(np.int64)


    final_result = {}
    for i, x in enumerate(detections_top_9['detection_classes']):
        print(i, x)
        #ymin, xmin, ymax, xmax = detections['detection_boxes'][6] # old one
        ymin, xmin, ymax, xmax = detections_top_9['detection_boxes'][i]
        im_width = image_np_with_detections.shape[1]
        im_height = image_np_with_detections.shape[0]

        #final_box.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])
        final_box = [xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]
        final_box = [int(x) for x in final_box]
        roi = image_np_with_detections[final_box[2]:final_box[3], final_box[0]:final_box[1]]

        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        os.environ['TESSDATA_PREFIX'] = r'/usr/share/tesseract-ocr/4.00/tessdata/'



        if map_for_class_name(int(x)) in ['sex', 'country', 'last_name_eng', 'name_eng']:
            tesseract_config = "-l eng --oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        else:
            tesseract_config = "-l digits --oem 1 --psm 13 -c tessedit_char_whitelist=0123456789"
        tesseract_data = pytesseract.image_to_string(roi, config=tesseract_config)
        tesseract_data = tesseract_data.replace("\n\x0c", "")
        class_name = map_for_class_name(int(x))
        if class_name == 'sex' and 'F' in tesseract_data: tesseract_data = 'F'
        if class_name == 'sex' and 'M' in tesseract_data: tesseract_data = 'M'
        if class_name in ['birth_day', 'exp_date'] and len(tesseract_data) == 8:
            tesseract_data = tesseract_data[:2] + '.' + tesseract_data[2:4] + '.' + tesseract_data[4:]

        if class_name not in ['name_geo', 'last_name_geo']:
            final_result[class_name] = tesseract_data

    return final_result


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def map_for_class_name(class_no):
    maping = {
        "1": "name_geo",
        "2": "name_eng",
        "3": "last_name_geo",
        "4": "last_name_eng",
        "5": "country",
        "6": "sex",
        "7": "id",
        "8": "birth_day",
        "9": "exp_date"
    }
    return maping[str(class_no)]
