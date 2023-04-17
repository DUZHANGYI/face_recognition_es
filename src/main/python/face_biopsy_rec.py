import random
import sys
import threading
import time

import cv2
import dlib
from imutils import face_utils

import face_detector
import face_recognizer
from config import biopsy_config
from utils import es_util, dlib_util

id2class = {0: 'No Mask', 1: 'Mask'}

WINK = 'wink'
MOUTH = 'mouth'
NOD = 'nod'
operates = [WINK, MOUTH, NOD]

# 储存每次截图的人脸画面 用于判断是否为同一人
FACE_ENCODINGS = []
# 是否开始进行下一次操作
next_operate = True

# 特征点检测器
predictor = dlib.shape_predictor(biopsy_config.landmarks_68_path)


def flush():
    global next_operate
    next_operate = len(FACE_ENCODINGS) < biopsy_config.gather_num


def liveness_detection():
    FACE_ENCODINGS.clear()
    blink_counter = 0
    # 初始化点头次数
    nod_total = 0
    head_counter = 0
    # 初始化眨眼次数
    blink_total = 0
    # 初始化张嘴次数
    mouth_total = 0
    # 初始化张嘴状态为闭嘴
    mouth_status_open = 0

    # 获取左眼的特征点
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # 获取右眼的特征点
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # 获取嘴巴特征点
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    process_this_frame = True

    name = None
    location = (0, 0, 0, 0)
    mask = None
    score = None

    is_operate_success = False
    op = None
    is_check = False

    font = cv2.FONT_HERSHEY_SIMPLEX

    known_names = []
    known_encodings = []

    vs = cv2.VideoCapture(0)
    while vs.isOpened():
        flag, frame = vs.read()
        if not flag:
            print("摄像头打开失败", flag)
            break
        if frame is None:
            continue

        # 图片转换成灰色（去除色彩干扰，让图片识别更准确）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        item = face_detector.detect(frame)
        if item:
            box, cls = item
            rec = dlib.rectangle(*box)
            shape = predictor(gray, rec)  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
            shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

            inner_mouth = shape[mStart:mEnd]  # 取出嘴巴对应的特征点
            mar = dlib_util.mouth_aspect_ratio(inner_mouth)  # 求嘴巴mar的均值

            left_eye = shape[lStart:lEnd]  # 取出左眼对应的特征点
            right_eye = shape[rStart:rEnd]  # 取出右眼对应的特征点
            left_ear = dlib_util.eye_aspect_ratio(left_eye)  # 计算左眼EAR
            right_ear = dlib_util.eye_aspect_ratio(right_eye)  # 计算右眼EAR
            ear = (left_ear + right_ear) / 2.0  # 求左右眼EAR的均值

            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), 1)

            global next_operate
            if next_operate and len(FACE_ENCODINGS) <= biopsy_config.gather_num:
                next_operate = False
                op = random.choice(list(operates))

            if op == WINK:
                if ear < biopsy_config.ear_thresh:
                    blink_counter += 1
                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                else:
                    if biopsy_config.ear_counter_frames_min <= blink_counter <= biopsy_config.ear_counter_frames_max:
                        blink_total += 1
                        print('捕捉到眨眼行为\n')
                        is_operate_success = True
                    blink_counter = 0
            # 通过张、闭来判断一次张嘴动作
            elif op == MOUTH:
                if mar > biopsy_config.mar_thresh:
                    mouth_status_open = 1
                else:
                    if mouth_status_open:
                        mouth_total += 1
                        print('捕捉到张嘴行为\n')
                        is_operate_success = True
                    mouth_status_open = 0
            elif op == NOD:
                # 获取头部姿态
                reproject_dst, euler_angle = dlib_util.get_head_pose(shape)
                har = euler_angle[0, 0]  # 取pitch旋转角度
                if har > biopsy_config.har_thresh:  # 点头阈值0.3
                    head_counter += 1
                else:
                    # 如果连续3次都小于阈值，则表示点头一次
                    if head_counter >= biopsy_config.nod_ar_counter_frames:
                        nod_total += 1
                        print('捕捉到点头行为\n')
                        is_operate_success = True
                    # 重置点头帧计数器
                    head_counter = 0

            if is_operate_success:
                op = None
                is_operate_success = False
                face_encoding = face_recognizer.get_img_face_encoding2(frame)
                FACE_ENCODINGS.append(face_encoding)
                threading.Timer(biopsy_config.heartbeat_interval, flush).start()

            if len(FACE_ENCODINGS) >= biopsy_config.gather_num and not is_check and process_this_frame:
                if len(known_names) <= 0 and len(known_encodings) <= 0:
                    start_time = time.time()
                    print('ES人脸比对开始')
                    face_encoding_list = face_recognizer.get_face_encoding_vector(frame)
                    known_names, known_encodings, scores = es_util.batch_search(face_encoding_list, min_score=0.94)
                    total_time = (time.time() - start_time) * 1000
                    print(f'ES人脸比对结束，总耗时：{total_time:.0f}毫秒')
                else:
                    item = face_recognizer.recognize(frame, known_names, known_encodings)
                    if item:
                        name, (left, top, right, bottom), mask, score = item
                        if score <= 0.5:
                            known_names.clear()
                            known_encodings.clear()
                        location = (left, top, right, bottom)

            (left, top, right, bottom) = location
            if mask is not None:
                cv2.putText(frame, id2class[mask], (100, 200), font, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                name_score = name + "  {:.2f}%".format(float(score * 100))
                cv2.putText(frame, name_score, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, op, (100, 100), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Mouth: {}".format(mouth_total), (100, 30), font, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Wink: {}".format(blink_total), (230, 30), font, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Nod: {}".format(nod_total), (330, 30), font, 0.7, (255, 0, 0), 2)

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)

        # 保持画面的持续。
        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()
    vs.release()
    sys.exit(0)


if __name__ == '__main__':
    liveness_detection()
