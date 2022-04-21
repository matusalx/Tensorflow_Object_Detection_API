# USAGE
# opencv-text-detection --image images/lebron_james.jpg
import os, sys
from os.path import dirname
project_dir = dirname(dirname(dirname(dirname(dirname(os.path.abspath(__file__))))))
sys.path.append(project_dir + r'/workplace/rotation/tomhoag_opencv_text_detection')
#print('test')
#print(project_dir + r'\workplace\rotation\tomhoag_opencv_text_detection')
import argparse
import os
import time
import cv2
from nms import nms
import numpy as np



from opencv_text_detection import utils
from opencv_text_detection.decode import decode

def text_detection(pil_image):
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    width = 640
    height = 640
    min_confidence = 0.9
    east = project_dir + r'/workplace/rotation/frozen_east_text_detection.pb'
    (origHeight, origWidth) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    ratioWidth = origWidth / float(newW)
    ratioHeight = origHeight / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (imageHeight, imageWidth) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (imageWidth, imageHeight), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
    # NMS on the the unrotated rects
    confidenceThreshold = min_confidence
    nmsThreshold = 0.4

    # decode the blob info
    (rects, confidences, baggage) = decode(scores, geometry, confidenceThreshold)

    offsets = []
    thetas = []
    for b in baggage:
        offsets.append(b['offset'])
        thetas.append(b['angle'])


    functions = [nms.fast.nms]
    # convert rects to polys
    polygons = utils.rects2polys(rects, thetas, offsets, ratioWidth, ratioHeight)

    print("[INFO] Running nms.polygons . . .")

    for i, function in enumerate(functions):

        start = time.time()
        indicies = nms.polygons(polygons, confidences, nms_function=function, confidence_threshold=confidenceThreshold,
                                nsm_threshold=nmsThreshold)
        end = time.time()

        indicies = np.array(indicies).reshape(-1)

        #drawpolys = np.array(polygons)[indicies]
        from math import pi  #test angle
        try:
            thetas = np.array(thetas)[indicies]

            degrees = thetas * 180 / pi
            print(np.sort(degrees))
            print(np.average(degrees))
            angle = int(np.average(degrees))
        except:
            angle=-1234
    return angle


