import os
import numpy as np
import pandas as pd
import cv2
import base64
from pathlib import Path
from PIL import Image
import requests
from detectors import FaceDetector

import tensorflow as tf

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])
from keras import utils


def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def load_image(img):
    exact_image = False
    base64_img = False
    url_img = False

    if type(img).__module__ == np.__name__:
        exact_image = True
    elif len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True
    elif len(img) > 11 and img.startswith("http"):
        url_img = True

    if base64_img:
        img = loadBase64Img(img)
    elif url_img:
        img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))
    elif not exact_image:  # image path passed as input
        if not os.path.isfile(img):
            raise ValueError("Confirm that ", img, " exists")
        img = cv2.imread(img)
    return img


def initialize_input(img1_path, img2_path):
    bulkProcess = False

    img_list = [[img1_path, img2_path]]

    return img_list, bulkProcess


def preprocess_face(img, target_size=(224, 224), grayscale=False, enforce_detection=True, detector_backend='opencv',
                    return_region=False, align=True):
    # img might be path, base6' or numpy array. Convert it to numpy whatever it is.
    img = load_image(img)

    base_img = img.copy()

    img, region = detect_face(img=img, detector_backend=detector_backend, grayscale=grayscale,
                              enforce_detection=enforce_detection, align=align)

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # post-processing
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize image to expected shape
    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if not grayscale:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # normalizing the image pixels
    img_pixels = utils.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    if return_region:
        return img_pixels, region
    else:
        return img_pixels


def detect_face(img, detector_backend='opencv', grayscale=False, enforce_detection=True, align=True):
    img_region = [0, 0, img.shape[0], img.shape[1]]
    # skip detection and alignment if they already have pre-processed images
    if detector_backend == 'skip':
        return img, img_region

    # detector stored in a global variable in FaceDetector object.
    # this call should be completed very fast because it will return found in memory
    # it will not build face detector model in each call (consider for loops)
    face_detector = FaceDetector.build_model(detector_backend)
    try:
        detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, img, align)
    except:  # if detected face shape is (0, 0) and alignment cannot be performed, this block will be run
        detected_face = None
    if isinstance(detected_face, np.ndarray):
        return detected_face, img_region
    else:
        if detected_face is None:
            if not enforce_detection:
                return img, img_region
            else:
                raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or "
                                 "consider to set enforce_detection param to False.")


def find_input_shape(model):
    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    # issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
    # whereas its older versions expect (y, x)
    if tf_major_version == 2 and tf_minor_version >= 5:
        x = input_shape[0];
        y = input_shape[1]
        input_shape = (y, x)

    if type(input_shape) == list:  # issue 197: some people got array here instead of tuple
        input_shape = tuple(input_shape)

    return input_shape


def normalize_input(img, normalization='base'):
    if normalization == 'base':
        return img
