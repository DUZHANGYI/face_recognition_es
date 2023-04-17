import math

import cv2
import dlib
import numpy as np

from config import biopsy_config

# 世界坐标系(UVW)：填写3D参考点
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角
# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_effective = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reproject_src = np.float32([[10.0, 10.0, 10.0],
                            [10.0, 10.0, -10.0],
                            [10.0, -10.0, -10.0],
                            [10.0, -10.0, 10.0],
                            [-10.0, 10.0, 10.0],
                            [-10.0, 10.0, -10.0],
                            [-10.0, -10.0, -10.0],
                            [-10.0, -10.0, 10.0]])

predictor = dlib.shape_predictor(biopsy_config.landmarks_68_path)


# 头部姿态估计
def get_head_pose(shape):
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_effective与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_effective)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reproject_dst, _ = cv2.projectPoints(reproject_src, rotation_vec, translation_vec, cam_matrix, dist_effective)
    reproject_dst = tuple(map(tuple, reproject_dst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    return reproject_dst, euler_angle  # 投影误差，欧拉角


# 眼长宽比例
def eye_aspect_ratio(eye):
    # (|e1-e5|+|e2-e4|) / (2|e0-e3|)
    a = np.linalg.norm(eye[1] - eye[5])
    b = np.linalg.norm(eye[2] - eye[4])
    c = np.linalg.norm(eye[0] - eye[3])
    ear = (a + b) / (2.0 * c)
    return ear


# 嘴长宽比例
def mouth_aspect_ratio(mouth):
    a = np.linalg.norm(mouth[1] - mouth[7])  # 61, 67
    b = np.linalg.norm(mouth[3] - mouth[5])  # 63, 65
    c = np.linalg.norm(mouth[0] - mouth[4])  # 60, 64
    mar = (a + b) / (2.0 * c)
    return mar


# 人脸对齐
def face_alignment(face_img):
    rec = dlib.rectangle(0, 0, face_img.shape[0], face_img.shape[1])
    shape = predictor(np.uint8(face_img), rec)
    # left eye, right eye, nose, left mouth, right mouth
    order = [36, 45, 30, 48, 54]
    for j in order:
        x = shape.part(j).x
        y = shape.part(j).y
    # 计算两眼的中心坐标
    eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
    dx = (shape.part(45).x - shape.part(36).x)
    dy = (shape.part(45).y - shape.part(36).y)
    # 计算角度
    angle = math.atan2(dy, dx) * 180. / math.pi
    # 计算仿射矩阵
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    # 进行仿射变换，即旋转
    rot_img = cv2.warpAffine(face_img, rotate_matrix, (face_img.shape[0], face_img.shape[1]))
    return rot_img


# 人脸修剪
def face_prune(img, box, size=128):
    # 识别人脸位置
    (left, top, right, bottom) = box
    x1 = top if top > 0 else 0
    y1 = bottom if bottom > 0 else 0
    x2 = left if left > 0 else 0
    y2 = right if right > 0 else 0
    face = img[x1:y1, x2:y2]
    # 调整图片的尺寸
    face = cv2.resize(face, (size, size))
    face = face_alignment(face)
    return face
