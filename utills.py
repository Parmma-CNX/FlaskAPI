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
import os
import random
from keras.applications import imagenet_utils
from skimage.io import imread
from skimage.transform import resize


ALLOWED_EXTENSIONS = set(['jpg'])


def allowed_file(filename):
    """Only .jpg files allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


def image_classification(image_claasifier):
    CATEGORIES = ["ShirtLongSleeve", "ShirtShortSleeve", "TshirtRneckLongSleeve",
                  "TshirtRshort", "TshirtVlongSleeve", "TshirtVneckShort"]
    image_claasifier = keras.models.model_from_json(json_config)
    image_claasifier.load_weights('model.h5')
    prediction = image_claasifier.predict_classes([prepare('tshirttest.jpg')])
    guess = CATEGORIES[int(prediction[0])]
    print(guess)
    return guess
