import os
from config.config_Faster_RCNN import cfg
from models.Faster_RCNN import Faster_RCNN
import cv2
import numpy as np
import time
from utils.faster_rcnn.anchors import all_anchor_conner, anchor_labels_process


def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a

    return y

# def py_cpu_nms(dets, scores, thresh):
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     # scores = dets[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]
#     return keep

def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def letterbox_image_cv2(image, w, h):
    ih = image.shape[0]
    iw = image.shape[1]

    scale = min(w / iw, h / ih)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # top bottom left right
    top_pad = int((h - image.shape[0]) / 2)
    bottom_pad = h - image.shape[0] - top_pad
    left_pad = int((w - image.shape[1]) / 2)
    right_pad = w - image.shape[1] - left_pad
    # print('top_pad, bottom_pad, left_pad, right_pad : ', top_pad, bottom_pad, left_pad, right_pad)
    pad_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                   value=[128, 128, 128])

    # print(pad_image.shape)
    # cv2.imshow('adf', pad_image)
    # cv2.waitKeyEx(0)
    return pad_image, scale


def coord_transform_inv(anchors, boxes):
    # anchors = anchors.astype(np.float32)
    anchors = np.reshape(anchors, [-1, 4])
    anchor_x = (anchors[:, 2] + anchors[:, 0]) * 0.5
    anchor_y = (anchors[:, 3] + anchors[:, 1]) * 0.5
    acnhor_w = (anchors[:, 2] - anchors[:, 0]) + 1.0
    acnhor_h = (anchors[:, 3] - anchors[:, 1]) + 1.0
    boxes = np.reshape(boxes, [-1, 4])
    boxes_x = boxes[:, 0] * acnhor_w + anchor_x
    boxes_y = boxes[:, 1] * acnhor_h + anchor_y
    boxes_w = np.exp(boxes[:, 2]) * acnhor_w
    boxes_h = np.exp(boxes[:, 3]) * acnhor_h
    coord_x1 = boxes_x - boxes_w * 0.5
    coord_y1 = boxes_y - boxes_h * 0.5
    coord_x2 = boxes_x + boxes_w * 0.5
    coord_y2 = boxes_y + boxes_h * 0.5
    coord_result = np.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)
    print(coord_x1, coord_y1, coord_x2, coord_y2)
    return coord_result


if __name__ == '__main__':
    input_dir = 'F:\\kwon\\coco_2017\\val2017\\val2017'
    files = os.listdir(input_dir)
    net = Faster_RCNN('Faster_RCNN', 'resnet101', False)

    for file in files:
        image = cv2.imread(os.path.join(input_dir, file))
        image = image[:, :, (2, 1, 0)]
        height, width = image.shape[:2]
        min = np.min([height, width])
        max = np.max([height, width])
        scale_min = cfg.min_image_size / min
        scale_max = cfg.max_image_size / max

        # 먄악 image를 scale로 키웠을 대 다른 쪽이 max보다 큰 경우 있으므로
        # scale = cfg.min_image_size / min
        if scale_min * max > cfg.max_image_size:
            scale = scale_max
        else:
            scale = scale_min
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        new_height, new_width = image.shape[:2]
        # all_anchor_conner(image_width, image_height, anchor_scales, anchor_ratios, stride=16):
        anchors = all_anchor_conner(new_width, new_height, cfg.anchor_scales, cfg.anchor_ratios, cfg.feat_stride)
        # image = image[:, :, (2, 1, 0)]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_image = image - [122.7717, 115.9465, 102.9801]
        input_image /= 128
        input_image = np.expand_dims(input_image, axis=0)

        feed_dict = {
            net.image: input_image,
            net.image_height: [new_height],
            net.image_width: [new_width],
            net.anchors: anchors
        }
        cls, bbox, nms_idx = net.sess.run(
            [net.cls_score, net.bbox_score, net.nms_idx], feed_dict=feed_dict)

        anchors = anchors[nms_idx]
        # cls = softmax(cls)
        pred_box_score_arg = np.argmax(cls, axis=1)
        num_pred = pred_box_score_arg.shape[0]
        pred_box_gather = np.empty([num_pred, 4], dtype=np.float32)
        pred_score_gather = np.empty(num_pred)

        for j in range(num_pred):
            pred_box_gather[j, :] = bbox[j, 4 * pred_box_score_arg[j]:4 * (pred_box_score_arg[j] + 1)]
            pred_score_gather[j] = cls[j, pred_box_score_arg[j]]

        pred_box_gather = pred_box_gather * np.array(cfg.bbox_nor_stdv) + np.array(cfg.bbox_nor_mean)
        print(pred_box_gather)

        anchor_x = (anchors[:, 2] + anchors[:, 0]) * 0.5
        anchor_y = (anchors[:, 3] + anchors[:, 1]) * 0.5
        acnhor_w = (anchors[:, 2] - anchors[:, 0]) + 1.0
        acnhor_h = (anchors[:, 3] - anchors[:, 1]) + 1.0

        prdict_x = pred_box_gather[:, 0] * acnhor_w + anchor_x
        prdict_y = pred_box_gather[:, 1] * acnhor_h + anchor_y
        prdict_w = np.exp(pred_box_gather[:, 2]) * acnhor_w
        prdict_h = np.exp(pred_box_gather[:, 3]) * acnhor_h

        # model이 예측한 bbox의 좌표(x1, y1, x2, y2)
        predcit_x1 = prdict_x - prdict_w * 0.5
        predcit_y1 = prdict_y - prdict_h * 0.5
        predcit_x2 = prdict_x + prdict_w * 0.5
        predcit_y2 = prdict_y + prdict_h * 0.5
        print(predcit_x1, predcit_y1, predcit_x2, predcit_y2)
        pre_box_coord = np.stack([predcit_x1, predcit_y1, predcit_x2, predcit_y2], axis=1)

        b0 = np.maximum(np.minimum(pre_box_coord[:, 0], (new_width - 1.0)), 0.0)
        b1 = np.maximum(np.minimum(pre_box_coord[:, 1], (new_height - 1.0)), 0.0)
        b2 = np.maximum(np.minimum(pre_box_coord[:, 2], (new_width - 1.0)), 0.0)
        b3 = np.maximum(np.minimum(pre_box_coord[:, 3], (new_height - 1.0)), 0.0)
        pre_box_coord = np.stack([b0, b1, b2, b3], axis=1)

        for k in range(1, cfg.num_classes):
            pre_class_arg = np.where(pred_box_score_arg == k)[0]
            print('pre_class_arg : ', pre_class_arg)
            cls_pred_box_coord = pre_box_coord[pre_class_arg, :]
            cls_pred_score = pred_score_gather[pre_class_arg]
            # print(cls_pred_box_coord.shape, cls_pred_score.shape)
            cls_pred_score = cls_pred_score[:, np.newaxis]
            cls_pred_target = np.concatenate((cls_pred_box_coord, cls_pred_score), axis=1)
            keep = py_cpu_nms(cls_pred_target, cfg.test_nms_thresh)
            cls_pred_target = cls_pred_target[keep, :]
            # print('cls_pred_target : ', cls_pred_target)
            print(cls_pred_target.shape)

            for coord in cls_pred_box_coord:
                print(int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3]))
                cv2.rectangle(image, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 255, 0))
                # cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))

        cv2.imshow('a', image[:, :, (2 ,1 ,0)])
        cv2.waitKey(0)



        # print('cls.shape : ', cls.shape)
        # print('bbox.shape : ', bbox.shape)
        # print('anchors.shape : ', anchors.shape)
        #
        # cls = softmax(cls)
        # arg_idx = np.argmax(cls, axis=1)
        # cls = np.max(cls, axis=1)
        # # print(arg_idx.shape)
        # # print(arg_idx)
        # # cls = cls[arg_idx, :]
        # print('cls.shape : ', cls.shape)
        #
        # # thresh_idx = score > 0.1
        # # print('thresh_idx.shape : ', thresh_idx.shape)
        # # print(cls[thresh_idx])
        # # print(bbox[thresh_idx])
        #
        # # score = score[thresh_idx]
        # # bbox = bbox[thresh_idx]
        #
        # print('bbox : ', bbox.shape)
        # print('anchors : ', anchors.shape)
        # print('bbox : ', bbox)
        #
        # anchor_x = (anchors[:, 2] + anchors[:, 0]) * 0.5
        # anchor_y = (anchors[:, 3] + anchors[:, 1]) * 0.5
        # acnhor_w = (anchors[:, 2] - anchors[:, 0]) + 1.0
        # acnhor_h = (anchors[:, 3] - anchors[:, 1]) + 1.0
        #
        # prdict_x = bbox[:, 0] * acnhor_w + anchor_x
        # prdict_y = bbox[:, 1] * acnhor_h + anchor_y
        # prdict_w = np.exp(bbox[:, 2]) * acnhor_w
        # prdict_h = np.exp(bbox[:, 3]) * acnhor_h
        #
        # # model이 예측한 bbox의 좌표(x1, y1, x2, y2)
        # predcit_x1 = prdict_x - prdict_w * 0.5
        # predcit_y1 = prdict_y - prdict_h * 0.5
        # predcit_x2 = prdict_x + prdict_w * 0.5
        # predcit_y2 = prdict_y + prdict_h * 0.5
        # predict_coord = np.stack([predcit_x1, predcit_y1, predcit_x2, predcit_y2], axis=1)
        #
        # # print(predict_coord)
        # # predict_coord = predict_coord[thresh_idx]
        # # score = score[thresh_idx]
        #
        # b0 = np.maximum(np.minimum(predict_coord[:, 0], (new_width - 1.0)), 0.0)
        # b1 = np.maximum(np.minimum(predict_coord[:, 1], (new_height - 1.0)), 0.0)
        # b2 = np.maximum(np.minimum(predict_coord[:, 2], (new_width - 1.0)), 0.0)
        # b3 = np.maximum(np.minimum(predict_coord[:, 3], (new_height - 1.0)), 0.0)
        # predict_coord = np.stack([b0, b1, b2, b3], axis=1)
        # print('predict_coord : ', predict_coord.shape)
        # idx = py_cpu_nms(predict_coord, cls, 0.5)
        # print('predict_coord : ', predict_coord.shape)
        # predict_coord = predict_coord[idx]
        # print('predict_coord : ', predict_coord.shape)
        #
        # for coord in predict_coord:
        #     print(int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3]))
        #     cv2.rectangle(image, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 255, 0))
        #     # cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
        #
        # cv2.imshow('a', image[:, :, (2 ,1 ,0)])
        # cv2.waitKey(0)