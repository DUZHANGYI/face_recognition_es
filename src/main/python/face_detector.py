import cv2
import numpy as np
import onnxruntime

from config import biopsy_config
from utils.detector_util import generate_anchors, decode_bbox, single_class_non_max_suppression

# load the model
ort_session = onnxruntime.InferenceSession(biopsy_config.onnxruntime_path)
# anchor configuration
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)


# 输入图片(numpy),输出人脸框体最大的box
def detect(image, conf_thresh=0.6, iou_thresh=0.4, target_shape=(360, 360)):
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    image_transposed = image_exp.transpose((0, 3, 1, 2)).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: image_transposed}
    y_bboxes_output, y_cls_output = ort_session.run(None, ort_inputs)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    # keep_idx is the alive bounding box after nms.
    keep_ids = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh, iou_thresh)
    max_area, r_item = -1, None
    for idx in keep_ids:
        # conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        x_min = max(0, int(bbox[0] * width))
        y_min = max(0, int(bbox[1] * height))
        x_max = min(int(bbox[2] * width), width)
        y_max = min(int(bbox[3] * height), height)
        item = (x_min, y_min, x_max, y_max), class_id
        area = (x_max - x_min) * (y_max - y_min)
        if max_area < area:
            max_area, r_item = area, item
    return r_item
