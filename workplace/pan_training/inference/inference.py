import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf


tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = r'workplace\pan_training\exported-models\my_faster_rcnn_1024\saved_model'
PATH_TO_SAVED_MODEL = r'workplace\pan_training\exported_models\my_faster_rcnn_1024_first\saved_model'
PATH_TO_LABELS = r'workplace\pan_training\annotations\pan_label_map.pbtxt'

print('Loading model...', end='')
start_time = time.time()
# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

path = r'workplace\pan_training\images\test\128.jpg'

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

IMAGE_PATHS = []
IMAGE_PATHS.append(path)

for image_path in IMAGE_PATHS:

    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    print('Done')
plt.show()


#image_np_with_detections[0.40016806:0.4787321,0.36464065:0.3836028]


test = image_np_with_detections
test1 = Image.fromarray(test)
test1.show()

test = image_np
test1 = Image.fromarray(test)
test1.show()


detections = detections[:10]

ymin, xmin, ymax, xmax =  detections['detection_boxes'][6]
im_width = image_np_with_detections.shape[1]
im_height = image_np_with_detections.shape[0]

#final_box = []
#final_box.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])
final_box = [xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height]

final_box = [int(x) for x in final_box]

test = image_np_with_detections[final_box[2]:final_box[3], final_box[0]:final_box[1]]
test1 = Image.fromarray(test)
test1.show()

test = image_np[final_box[2]:final_box[3], final_box[0]:final_box[1]]
test1 = Image.fromarray(test)
test1.show()

'''
If we consider (0,0) as top left corner of image called im with 
left-to-right as x direction and top-to-bottom as y direction. 
and we have (x1,y1) as the top-left vertex and (x2,y2) as the bottom-right vertex of a rectangle
region within that image, then:

roi = im[y1:y2, x1:x2]
'''


import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\v.gurgenishvili\Tesseract-OCR\tesseract.exe'

eng_config = "-l eng --oem 1 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
digit_config = "-l digits --oem 1 --psm 13 -c tessedit_char_whitelist=0123456789"

text = pytesseract.image_to_string(test1, config=eng_config)

digits = pytesseract.image_to_string(test1, config=digit_config)




