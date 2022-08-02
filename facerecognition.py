import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from basemodels import VGGFace
from commons import functions

from commons import functions, distance as dst


def represent(img_path, model=None, enforce_detection=True, detector_backend='opencv',
              align=True, normalization='base'):
    """
    This function represents facial images as vectors

    Parameters:
		img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.
		model: Built model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.
			model = build_model('VGG-Face')
		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.
		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
		normalization (string): normalize the input image before feeding to model
	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """

    if model is None:
        model = build_model('VGG-Face')
    # decide input shape
    input_shape_x, input_shape_y = functions.find_input_shape(model)

    # detect and align
    img = functions.preprocess_face(img=img_path, target_size=(input_shape_y, input_shape_x),
                                    enforce_detection=enforce_detection, detector_backend=detector_backend, align=align)
    # custom normalization
    img = functions.normalize_input(img=img, normalization=normalization)

    # represent
    embedding = model.predict(img)[0].tolist()

    return embedding


def build_model():
    """
    This function builds a VGG-Face model
    Returns:
    	built model
    """
    model_name = 'VGG-Face'
    global model_obj  # singleton design pattern

    models = {
        'VGG-Face': VGGFace.loadModel,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        else:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model_obj[model_name]


def verify(img1_path, img2_path, distance_metric='cosine', model=None, enforce_detection=True,
           detector_backend='opencv', align=True, prog_bar=True, normalization='base'):
    """
    This function verifies an image pair is same person or different persons.
    Parameters:
        img1_path, img2_path (string): exact image path, numpy array (BGR) or based64 encoded images could be passed.
        distance_metric (string): cosine, euclidean, euclidean_l2
        model (string): Built model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.
        enforce_detection (boolean): If no face could not be detected in an image, then this function will return exception by default. Set this to False not to have this exception. This might be convenient for low resolution images.
        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
        align (boolean): face alignment
        prog_bar (boolean): enable/disable a progress bar
        normalization (string): base: no normalization, ..
    Returns:
        Verify function returns a dictionary.
    """
    tic = time.time()

    img_list = [[img1_path, img2_path]]

    resp_objects = []

    model_names = []
    metrics = []
    model_names.append('VGG-Face')
    metrics.append(distance_metric)

    # model_name = 'VGG-Face'
    # metric = distance_metric
    if model is None:
        model = build_model()
    models = {}
    models['VGG-Face'] = model

    disable_option = (False if len(img_list) > 1 else True) or not prog_bar

    pbar = tqdm(range(0, len(img_list)), desc='Verification', disable=disable_option)

    for index in pbar:
        instance = img_list[index]

        if type(instance) == list and len(instance) >= 2:
            img1_path = instance[0];
            img2_path = instance[1]

            custom_model = models['VGG-Face']
            img1_representation = represent(img_path=img1_path
                                            , model=custom_model
                                            , enforce_detection=enforce_detection, detector_backend=detector_backend
                                            , align=align
                                            , normalization=normalization
                                            )
            img2_representation = represent(img_path=img2_path
                                            , model=custom_model
                                            , enforce_detection=enforce_detection, detector_backend=detector_backend
                                            , align=align
                                            , normalization=normalization
                                            )


            # find distance between embeddings
            for j in metrics:
                if j == 'cosine':
                    distance = dst.findCosineDistance(img1_representation, img2_representation)
                else:
                    raise ValueError('Invalid distance_metric passed - ', distance_metric)

                distance = np.float64(distance)

                # decision

                threshold = dst.findThreshold('VGG-Face', j)

                if distance <= threshold:
                    identified = True
                else:
                    identified = False

                resp_obj = {
                    "verified": identified,
                    "distance": distance,
                    "threshold": threshold,
                    "model": 'VGG-Face',
                    "detector_backend": detector_backend,
                    "similarity_metric": distance_metric
                }
                return resp_obj

        else:
            raise ValueError("Invalid arguments passed to verify function: ", instance)

    toc = time.time()

