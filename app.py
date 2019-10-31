import os
from flask import Flask, render_template, request, jsonify
from utills import allowed_file
import flask
import cv2
import os.path
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
# Helper libraries
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
import cv2
import pandas
from keras.applications import imagenet_utils
from skimage.io import imread
import skimage.transform as st
from skimage import io

with open('model_config.json') as json_file:
    json_config = json_file.read()
    image_claasifier = keras.models.model_from_json(json_config)
    image_claasifier.load_weights('model.h5')


def prepare(filepath):
    try:
        IMG_SIZE = 100  # 50 in txt-based
        img_array = cv2.imread(filepath)  # read in the image
        # resize image to match model's expected sizing
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        # return the image with shaping
        return new_array.reshape(1, 100, 100, 3)
    except Exception as e:
        print(str(e))


app = Flask(__name__)
# app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
# <bound method MultiDict.to_dict of ImmutableMultiDict([('testpicture', <FileStorage: 'tshirttest.jpg' ('image/jpeg')>)])>


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    CATEGORIES = ["ShirtLongSleeve", "ShirtShortSleeve", "TshirtRneckLongSleeve",
                  "TshirtRshort", "TshirtVlongSleeve", "TshirtVneckShort"]
    image_claasifier = keras.models.model_from_json(json_config)
    image_claasifier.load_weights('model.h5')
    # print('testpicture' in request.files)
    # data = dict( request.files)
    # img = data['testpicture']
    """Image Classification"""
    if request.method == 'POST' and 'testpicture' in request.files:
        image = request.files['testpicture']
        image_predict = io.imread(image)
        image_predict2 = st.resize(image_predict, (100, 100))
        image = image_predict2.reshape(1, 100, 100, 3)
        prediction = image_claasifier.predict_classes(image)
        guess = CATEGORIES[int(prediction[0])]
        return jsonify({'guess': guess})
    else:
        return jsonify({'error': "Please upload a .jpg file"})


if __name__ == "__main__":
    print("loading the payload...")
    print("starting server...")
    app.run(host="localhost")
