import os
import cv2
from keras.preprocessing.image import *
import keras
import numpy as np
from final_image_builder import *

def predict(images):
    size = 64
    i=0
    images_arr = np.zeros((len(images), size, size, 1))
    for image in images:
        temp_image = cv2.resize(image,(size,size))
        # convert the image pixels to a numpy array
        temp_image = keras.backend.cast_to_floatx(img_to_array(temp_image)) / 255
        images_arr[i, :, :, :] = temp_image
        i += 1
    seq_length = (len(images))
    model = keras.models.load_model('./image_classifier.h5')
    image_type_vec = model.predict(images_arr)
    image_type = np.round((np.sum(image_type_vec)/seq_length))
    if image_type == 0: # the entered sequence is of doc type
        # resize text images to 128x128
        size = 128
        i = 0
        images_arr = np.zeros((len(images), size, size, 1))
        for image in images:
            temp_image = cv2.resize(image,(size,size))
            # convert the image pixels to a numpy array
            temp_image = keras.backend.cast_to_floatx(img_to_array(temp_image)) / 255
            images_arr[i, :, :, :] = temp_image
            i += 1
    labels = build_image(images_arr, image_type)

    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)


    Y = predict(images)
    return Y
