# 活体检测配置文件
onnxruntime_path = '../resources/data/ssd_mini_w360.onnx'
landmarks_68_path = '../resources/data/shape_predictor_68_face_landmarks.dat'
landmarks_5_path = '../resources/data/shape_predictor_5_face_landmarks.dat'
resnet_model_path = '../resources/data/dlib_face_recognition_resnet_model_v1.dat'
# 点头阈值0.3
har_thresh = 0.3
nod_ar_counter_frames = 3
# 嘴长宽比例值
mar_thresh = 0.2
# 眼长宽比例值
ear_thresh = 0.25
# 当ear小于阈值时，接连多少帧一定发生眨眼动作
ear_counter_frames_min = 1
ear_counter_frames_max = 5
# 采集次数
gather_num = 4
# 检测间隔
heartbeat_interval = 1
