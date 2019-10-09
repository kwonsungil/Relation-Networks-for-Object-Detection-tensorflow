import numpy as np

def calc_iou(boxes1, boxes2):

    dy = np.min([boxes2[3], boxes1[3]]) - np.max([boxes2[1], boxes1[1]])
    dx = np.min([boxes2[2], boxes1[2]]) - np.max([boxes2[0], boxes1[0]])
    inter_square = dx * dy
    # 큰 좌표에서는 작읍값, 작은 좌표에서는 큰값
    square1 = (boxes1[3]-boxes1[1]) * (boxes1[2]-boxes1[0])
    square2 = (boxes2[3]-boxes2[1]) * (boxes2[2]-boxes2[0])
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    # print('dx,dy : ', dx,dy)
    # print('square1 : ', square1)
    # print('square2 : ', square2)
    # print('union_square : ', union_square)
    return np.clip(inter_square / union_square, .0, 1.)

def intersect(boxes1, boxes2):
    # top, left, bottom, right = boxes
    # ymin xmin ymax xmax

    return not (boxes1[3] < boxes2[1] or boxes1[1] > boxes2[3] or boxes1[0] > boxes2[2] or boxes1[2] < boxes2[0])