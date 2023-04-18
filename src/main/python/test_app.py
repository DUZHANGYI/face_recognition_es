import pickle

import cv2
import dlib
import numpy as np

import face_recognizer
from utils import es_util


def test_create_index():
    es_util.create_index()


def test_put_face():
    names_pkl_path = '../resources/db/face_names.pkl'
    encodings_pkl_path = '../resources/db/face_encodings.pkl'
    known_names = pickle.load(open(names_pkl_path, "rb"))
    known_encodings = pickle.load(open(encodings_pkl_path, "rb"))
    for encoding, name in zip(known_encodings, known_names):
        face_encoding = np.array(dlib.vector(encoding))
        es_util.put_face(name, face_encoding.tolist())


def test_search_face():
    face_img = cv2.imread('../resources/images/pyy.jpg')
    face_encoding_list = face_recognizer.get_face_encoding_vector(face_img)
    name, score = es_util.search(face_encoding_list)
    print(f"匹配人：{name}   相似度：{score}")
