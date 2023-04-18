import os

import cv2
import dlib
import numpy as np

import face_detector
from config import biopsy_config

sp = dlib.shape_predictor(biopsy_config.landmarks_5_path)
facerec = dlib.face_recognition_model_v1(biopsy_config.resnet_model_path)


def _get_face_feat(img_x, box):
    rec = dlib.rectangle(*box)
    shape = sp(img_x, rec)
    return facerec.compute_face_descriptor(img_x, shape)


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)
    # 转为数组
    face_encodings_np = np.asarray(face_encodings)
    # 计算欧式距离
    return np.linalg.norm(face_encodings_np - face_to_compare, axis=1)


def get_img_face_encoding(fpath):
    img_x = cv2.imread(fpath)
    return get_img_face_encoding2(img_x)


def get_face_encoding_vector(img):
    face_encoding = get_img_face_encoding2(img)
    if face_encoding is None:
        return []
    face_encoding = np.array(dlib.vector(face_encoding))
    return face_encoding.tolist()


def get_img_face_encoding2(img):
    img_x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    item = face_detector.detect(img_x)
    if item:
        box, _ = item
        return _get_face_feat(img_x, box)


def create_known_faces(root):
    _known_faces = []
    _know_name = []
    for i, file in enumerate(os.listdir(root)):
        fpath = os.path.join(root, file)
        face_encoding = get_img_face_encoding(fpath)
        name = file.split('.')[0]
        # 检查 face_encoding 是否是 dlib.vector 类型的实例
        if isinstance(face_encoding, dlib.vector):
            _known_faces.append(face_encoding)
            _know_name.append(name)
        else:
            print(name + " is not a dlib vector")

    return _know_name, _known_faces


def recognize_all_match(image, _known_faces):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    item = face_detector.detect(image)
    if item:
        box, cls = item
        face_feat = _get_face_feat(image, box)
        scores = 1 - face_distance(_known_faces, face_feat)
        ix = np.argmax(scores).item()
        return box, 1 - int(cls), scores[ix]


def recognize(image, _know_name, _known_faces):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    item = face_detector.detect(image)
    if item:
        box, cls = item
        face_feat = _get_face_feat(image, box)
        scores = 1 - face_distance(_known_faces, face_feat)
        ix = np.argmax(scores).item()
        return _know_name[ix], box, 1 - int(cls), scores[ix]
