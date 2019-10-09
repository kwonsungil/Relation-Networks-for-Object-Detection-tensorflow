from config.config_Faster_RCNN import cfg
import numpy as np
import numpy.random as npr
from utils.faster_rcnn.anchors import calculate_IOU


# def calculate_IOU(target_boxes, gt_boxes):  # gt_boxes[num_obj,4] targer_boxes[w*h*k,4]
#     num_gt = gt_boxes.shape[0]
#     num_tr = target_boxes.shape[0]
#     IOU_s = np.zeros((num_gt, num_tr), dtype=np.float)
#     for ix in range(num_gt):
#         gt_area = (gt_boxes[ix, 2] - gt_boxes[ix, 0]) * (gt_boxes[ix, 3] - gt_boxes[ix, 1])
#         # print (gt_area)
#         for iy in range(num_tr):
#             iw = min(gt_boxes[ix, 2], target_boxes[iy, 2]) - max(gt_boxes[ix, 0], target_boxes[iy, 0])
#             # print (iw)
#             if iw > 0:
#                 ih = min(gt_boxes[ix, 3], target_boxes[iy, 3]) - max(gt_boxes[ix, 1], target_boxes[iy, 1])
#                 # print (ih)
#                 if ih > 0:
#                     tar_area = (target_boxes[iy, 2] - target_boxes[iy, 0]) * (target_boxes[iy, 3] - target_boxes[iy, 1])
#                     # print (tar_area)
#                     i_area = iw * ih
#                     iou = i_area / float((gt_area + tar_area - i_area))
#                     IOU_s[ix, iy] = iou
#     IOU_s = np.transpose(IOU_s)
#     return IOU_s


def proposal_target(rois, rois_score, gt_boxes, _num_classes, gt_cls):
    num_images = 1
    rois_per_image = cfg.dect_train_batch / num_images
    # 배경이 아닌 것에 대한 비율
    # rois_per_image = 256
    # dect_fg_rate = 0.25
    # fg_rois_per_image
    fg_rois_per_image = np.round(cfg.dect_fg_rate * rois_per_image)
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = sample_rois( \
        rois, rois_score, gt_boxes, fg_rois_per_image, \
        rois_per_image, _num_classes, gt_cls)
    # rois = rois.reshape(-1, 5)
    rois = rois.reshape(-1, 4)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """ compute the bbox_targets and bbox_inside_weights
        ie, tx*,ty*,tw*,th* and which bbox_target to be used in loss compute"""
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.roi_input_inside_weight
    return bbox_targets, bbox_inside_weights


def sample_rois(rois, rois_score, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, gt_cls):
    """ rois sample process:  clip to the image boundary, nms, bg_fg sample"""
    # 300개의 RoI가 있고 5개의 GT 가 있을 때
    # (300, 5)
    # overlaps = calculate_IOU(rois[:, 1:5], gt_boxes)
    overlaps = calculate_IOU(rois, gt_boxes)
    # RoI에서 GT와 가장 크게 겹치는 박스
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    # gt cls는 5개의 object[0, 4, 6, 2, 9]
    labels = gt_cls[gt_assignment]
    # labels = 300개의 RoI 중에서 가장 큰 값에 argmax에 대한 cls 값을 가짐
    # 300개의 RoI가 각각 class label 을 갖게 됨


    # IOU가 0.5 이상인 애들
    fg_inds = np.where(max_overlaps >= cfg.fg_thresh)[0]
    # IOU가 0.5 보다 작고 0보다 큰 애들
    bg_inds = np.where((max_overlaps < cfg.bg_thresh_hi) & (max_overlaps >= cfg.bg_thresh_lo))[0]
    # print(np.sum(fg_inds), np.sum(bg_inds))
    if fg_inds.size > 0 and bg_inds.size > 0:
        # 일단 256 * 1/4 만틈 positive를 가져오지만 그게 안될 경우 nehative로 채움
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        # 만양 negative도 batch 보다 작다면 중복해서 batch를 만듦
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:
        # 만약 negative가 없을 경우 positive로 batch를 채움
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        # 만약 positive가 없을 경우 negative로 채움
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        import pdb
        pdb.set_trace()
    keep_inds = np.append(fg_inds, bg_inds)
    #labels은 300개의 target ROI에 대한 class 들
    labels = labels[keep_inds]
    # positive anchor가 아닌 label은 0으로 바꿈
    # 0 = background
    labels[int(fg_rois_per_image):] = 0
    # 모든 roi 중 bath에 들어가는 256 roi
    rois = rois[keep_inds]
    roi_scores = rois_score[keep_inds]
    # 256개 roi, gt_boxes는 x1, x2, y1 ,y2
    # 현재 bbox는 좌표 값임
    # bbox_target_data = compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :], labels)
    bbox_target_data = compute_targets(rois, gt_boxes[gt_assignment[keep_inds], :], labels)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


def compute_targets(ex_rois, gt_rois, labels):

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    # 이것은 아직 모르겠당
    if cfg.bbox_nor_target_pre:
        targets = ((targets - np.array(cfg.bbox_nor_mean)) / np.array(cfg.bbox_nor_stdv))
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def bbox_transform(ex_rois, gt_rois):
    """ convert the coordinate of gt_rois into targets form using ex_rois """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets
