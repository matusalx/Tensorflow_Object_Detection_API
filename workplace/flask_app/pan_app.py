import os, sys
from os.path import dirname

from flask import Flask, render_template, request, redirect, url_for
from workplace.pan_training.inference.inference_function import predict
from workplace.classification_for_vision.pytorch_classification import predict_rotation
from workplace.rotation.tomhoag_opencv_text_detection.opencv_text_detection.text_detection import text_detection
from PIL import Image
import time
project_dir = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    # if uploaded_file.filename != '':
    #     uploaded_file.save(uploaded_file.filename)
    start_time = time.time()

    uploaded_image = Image.open(uploaded_file)
    uploaded_image = uploaded_image.rotate(-text_detection(uploaded_image))
    uploaded_image = uploaded_image.rotate(-text_detection(uploaded_image))
    #test = {'angle': text_detection(uploaded_image)}
    test = predict(uploaded_image)
    if len(test) < 7:
        test = {'fail': 'need rotation'}
        uploaded_image = predict_rotation(uploaded_image)
        uploaded_image = uploaded_image.rotate(-text_detection(uploaded_image))
        uploaded_image = uploaded_image.rotate(-text_detection(uploaded_image))

        test = predict(uploaded_image)
        #test = {'class_no': class_no}
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! image processing {} seconds'.format(elapsed_time))

    return test, 200


if __name__ == '__main__':
    #start_time = time.time()
    app.run(host='0.0.0.0', port=5000, debug=False)
    #text_detection(1)
    #print(os.getcwd())
    #predict_rotation(1)
    #print (dirname(dirname(dirname(os.path.abspath(__file__))))
    #uploaded_image = Image.open('Capture.jpg')
    #test = predict(uploaded_image)
    #print(test)
    #end_time = time.time()
    #elapsed_time = end_time - start_time
    #print('Done! image processing {} seconds'.format(elapsed_time))

