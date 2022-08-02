import time
import os
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


def detectFace(img_path, target_size=(224, 224), detector_backend='opencv', enforce_detection=True, align=True):
    """
    This function applies pre-processing stages of a face recognition pipeline including detection and alignment

    Parameters:
        img_path: exact image path, numpy array (BGR) or base64 encoded image

        detector_backend (string): face detection backends are retinaface, mtcnn, opencv, ssd or dlib

    Returns:
        detected and aligned face in numpy format
    """
    img = functions.preprocess_face(img=img_path, target_size=target_size, detector_backend=detector_backend,
                                    enforce_detection=enforce_detection, align=align)[
        0]  # preprocess_face returns (1, 224, 224, 3)
    return img[:, :, ::-1]  # bgr to rgb


def find(img_path, db_path, distance_metric='cosine', model=None, enforce_detection=True, detector_backend='opencv',
         align=True, prog_bar=True, normalization='base', silent=False):
    """
    This function applies verification several times and find an identity in a database

    Parameters:
        img_path: exact image path, numpy array (BGR) or based64 encoded image. If you are going to find several identities, then you should pass img_path as array instead of calling find function in a for loop. e.g. img_path = ["img1.jpg", "img2.jpg"]
        db_path (string): You should store some .jpg files in a folder and pass the exact folder path to this.
        distance_metric (string): cosine, euclidean, euclidean_l2
        model: built deepface model. A face recognition models are built in every call of find function. You can pass pre-built models to speed the function up.
        enforce_detection (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.
        detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
        prog_bar (boolean): enable/disable a progress bar

    Returns:
        This function returns pandas data frame. If a list of images is passed to img_path, then it will return list of pandas data frame.
    """

    tic = time.time()

    img_paths = [img_path]

    # -------------------------------

    if os.path.isdir(db_path) is True:

        if model is None:

            model = build_model()
            models = {}
            models['VGG-Face'] = model

        else:  # model != None
            if not silent: print("Already built model is passed")

            models = {}
            models['VGG-Face'] = model

        # ---------------------------------------

        model_names = [];
        metric_names = []
        model_names.append('VGG-Face')
        metric_names.append(distance_metric)

        # ---------------------------------------

        file_name = "representations_%s.pkl" % ("VGG-Face")
        file_name = file_name.replace("-", "_").lower()

        if path.exists(db_path + "/" + file_name):

            if not silent: print("WARNING: Representations for images in ", db_path,
                                 " folder were previously stored in ", file_name,
                                 ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")

            f = open(db_path + '/' + file_name, 'rb')
            representations = pickle.load(f)

            if not silent: print("There are ", len(representations), " representations found in ", file_name)

        else:  # create representation.pkl from scratch
            employees = []

            for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
                for file in f:
                    if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                        exact_path = r + "/" + file
                        employees.append(exact_path)

            if len(employees) == 0:
                raise ValueError("There is no image in ", db_path,
                                 " folder! Validate .jpg or .png files exist in this path.")

            # ------------------------
            # find representations for db images

            representations = []

            pbar = tqdm(range(0, len(employees)), desc='Finding representations', disable=prog_bar)

            # for employee in employees:
            for index in pbar:
                employee = employees[index]

                instance = []
                instance.append(employee)

                custom_model = models['VGG-Face']

                representation = represent(img_path=employee
                                           , model=custom_model
                                           , enforce_detection=enforce_detection, detector_backend=detector_backend
                                           , align=align
                                           , normalization=normalization
                                           )

                instance.append(representation)

                # -------------------------------

                representations.append(instance)

            f = open(db_path + '/' + file_name, "wb")
            pickle.dump(representations, f)
            f.close()

            if not silent: print("Representations stored in ", db_path, "/", file_name,
                                 " file. Please delete this file when you add new identities in your database.")

        # ----------------------------
        # now, we got representations for facial database

        df = pd.DataFrame(representations, columns=["identity", "%s_representation" % ('VGG-Face')])

        df_base = df.copy()  # df will be filtered in each img. we will restore it for the next item.

        resp_obj = []

        global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing', disable=prog_bar)
        for j in global_pbar:
            img_path = img_paths[j]

            # find representation for passed image

            custom_model = models['VGG-Face']

            target_representation = represent(img_path=img_path
                                              , model=custom_model
                                              , enforce_detection=enforce_detection,
                                              detector_backend=detector_backend
                                              , align=align
                                              , normalization=normalization
                                              )

            for k in metric_names:
                distances = []
                for index, instance in df.iterrows():
                    source_representation = instance["%s_representation" % ('VGG-Face')]

                    if k == 'cosine':
                        distance = dst.findCosineDistance(source_representation, target_representation)
                    elif k == 'euclidean':
                        distance = dst.findEuclideanDistance(source_representation, target_representation)
                    elif k == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation),
                                                             dst.l2_normalize(target_representation))

                    distances.append(distance)

                # ---------------------------

                df["%s_%s" % ('VGG-Face', k)] = distances

                threshold = dst.findThreshold('VGG-Face', k)
                df = df.drop(columns=["%s_representation" % ('VGG-Face')])
                df = df[df["%s_%s" % ('VGG-Face', k)] <= threshold]

                df = df.sort_values(by=["%s_%s" % ('VGG-Face', k)], ascending=True).reset_index(drop=True)

                resp_obj.append(df)
                df = df_base.copy()  # restore df for the next iteration

        # ----------------------------------

        toc = time.time()

        if not silent: print("find function lasts ", toc - tic, " seconds")

        if len(resp_obj) == 1:
            return resp_obj[0]

        return resp_obj

    else:
        raise ValueError("Passed db_path does not exist!")

    return None
