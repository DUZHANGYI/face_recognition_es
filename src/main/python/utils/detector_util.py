import numpy as np


def decode_bbox(anchors, raw_outputs, variances=None):
    """
    Decode the actual bbox according to the anchors.
    the anchor value order is:[x_min,y_min, x_max, y_max]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    """
    if variances is None:
        variances = [0.1, 0.1, 0.2, 0.2]
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_x_min = predict_center_x - predict_w / 2
    predict_y_min = predict_center_y - predict_h / 2
    predict_x_max = predict_center_x + predict_w / 2
    predict_y_max = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_x_min, predict_y_min, predict_x_max, predict_y_max], axis=-1)
    return predict_bbox


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios):
    """
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :return:
    """
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0]
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    """
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    """
    if len(bboxes) == 0:
        return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    x_max = bboxes[:, 2]
    y_max = bboxes[:, 3]

    area = (x_max - x_min + 1e-3) * (y_max - y_min + 1e-3)
    idx_sort = np.argsort(confidences)

    while len(idx_sort) > 0:
        last = len(idx_sort) - 1
        i = idx_sort[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_x_min = np.maximum(x_min[i], x_min[idx_sort[:last]])
        overlap_y_min = np.maximum(y_min[i], y_min[idx_sort[:last]])
        overlap_x_max = np.minimum(x_max[i], x_max[idx_sort[:last]])
        overlap_y_max = np.minimum(y_max[i], y_max[idx_sort[:last]])
        overlap_w = np.maximum(0, overlap_x_max - overlap_x_min)
        overlap_h = np.maximum(0, overlap_y_max - overlap_y_min)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idx_sort[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idx_sort = np.delete(idx_sort, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    return conf_keep_idx[pick]
