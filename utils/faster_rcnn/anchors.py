import numpy as np
# from utils.faster_rcnn.utils_an import intersect, calc_iou

def get_anchor_size(anchor_area, anchor_ratios):
    width = np.round(np.sqrt(anchor_area/anchor_ratios))
    length = width * anchor_ratios
    anchors = np.stack((width, length), axis=-1)
    return anchors


def ratios_process(anchor_scales, anchor_ratios):
    anchor_area = anchor_scales[:,0] * anchor_scales[:,1]
    anchors = np.vstack([get_anchor_size(anchor_area[i], anchor_ratios) for i in range(anchor_area.shape[0])])
    return anchors

def generate_anchors(anchor_scales, anchor_ratios, anchor_bias_x_ctr=8, anchor_bias_y_ctr=8):
    anchor_width = np.array(anchor_scales)
    anchor_length = np.array(anchor_scales)
    anchor_ratios = np.array(anchor_ratios)
    bias_x_ctr = anchor_bias_x_ctr
    bias_y_ctr = anchor_bias_y_ctr
    anchor_scales = np.stack((anchor_width, anchor_length), axis=-1)
    anchor_size = ratios_process(anchor_scales, anchor_ratios)
    # 1:1 짜리 128 256 512, 1:2 짜리 ....
    anchor_conner = generate_anchors_conner(anchor_size, bias_x_ctr, bias_y_ctr)
    return anchor_conner

def generate_anchors_conner(anchor_size, x_ctr, y_ctr):
    width = anchor_size[:,0]
    length = anchor_size[:,1]
    x1 = np.round(x_ctr - 0.5*width)
    y1 = np.round(y_ctr -0.5*length)
    x2 = np.round(x_ctr + 0.5*width)
    y2 = np.round(y_ctr +0.5*length)
    conners = np.stack((x1, y1, x2, y2), axis=-1)
    return conners


def all_anchor_conner(image_width, image_height, anchor_scales, anchor_ratios, stride=16):
    bias_anchor_conner = generate_anchors(anchor_scales, anchor_ratios)
    stride = np.float32(stride)
    dmap_width = np.ceil(image_width/stride)
    dmap_height = np.ceil(image_height/stride)
    # dmap_width = round(image_width/stride)
    # dmap_height = round(image_height/stride)
    # total_pos = int(dmap_height*dmap_width).astype(np.int32)
    total_pos = int(dmap_height*dmap_width)
    #offset_x = tf.range(dmap_width) * stride
    #offset_y = tf.range(dmap_height) * stride
    offset_x = np.arange(dmap_width) * stride
    offset_y = np.arange(dmap_height) * stride
    #x,y = tf.meshgrid(offset_x,offset_y)
    # print('offset_x,offset_y : ', offset_x, offset_y)
    x,y = np.meshgrid(offset_x,offset_y)
    x = np.reshape(x, [-1])
    y = np.reshape(y, [-1])
    coordinate = np.stack((x, y, x, y), axis=-1)
    # print('coordinate : ', coordinate)
    #coordinate = tf.reshape(coordinate, [total_pos,1,4])
    #coordinate = tf.reshape(coordinate, [total_pos,4])
    coordinate= np.transpose(np.reshape(coordinate, [1, total_pos, 4]), (1, 0, 2))
    # print('coordinate : ', coordinate[-3 :])
    # print (coordinate.shape)
    # print('bias_anchor_conner : ', bias_anchor_conner)
    # print (bias_anchor_conner.shape)
    all_anchor_conners = coordinate + bias_anchor_conner
    # print('all_anchor_conners : ', all_anchor_conners)
    # print('all_anchor_conners.shape : ', all_anchor_conners.shape)
    all_anchor_conners = np.reshape(all_anchor_conners, [-1, 4])
    # print('all_anchor_conners.shape : ', all_anchor_conners.shape)
    # print('all_anchor_conners : ', all_anchor_conners)
    return np.array(all_anchor_conners).astype(np.float32)


def calc_iou(boxes1, boxes2):

    dx = np.min([boxes2[3], boxes1[3]]) - np.max([boxes2[1], boxes1[1]])
    dy = np.min([boxes2[2], boxes1[2]]) - np.max([boxes2[0], boxes1[0]])
    inter_square = dx * dy
    # 큰 좌표에서는 작읍값, 작은 좌표에서는 큰값
    square1 = (boxes1[3]-boxes1[1]) * (boxes1[2]-boxes1[0])
    square2 = (boxes2[3]-boxes2[1]) * (boxes2[2]-boxes2[0])
    union_square = np.maximum(square1 + square2 - inter_square, 1e-10)
    return np.clip(inter_square / union_square, .0, 1.)

def intersect(boxes1, boxes2):
    # 1. box2 ymin이 box1의 ymax 보다 클 경우
    # 2. box2의 ymax가 box1의 ymin 보다 작은 경우
    # 3. box2의 xmax가 box1의 xmin 보다 작은 경우
    # 4. box2의 xmin이 box1의 xmax보다 큰 경우
    # 간단하게 index 작은 쪽이 큰 경우 (4가지) 만 아니면 겹친다
    return not (boxes1[3] < boxes2[1] or boxes1[1] > boxes2[3] or boxes1[0] > boxes2[2] or boxes1[2] < boxes2[0])

# def calculate_IOU (target_boxes, gt_boxes):
#     num_gt = gt_boxes.shape[0]
#     num_tr = target_boxes.shape[0]
#     IOU_s = np.zeros((num_gt, num_tr), dtype=np.float)
#     for ix in range(num_gt):
#         gt_area = (gt_boxes[ix,2]-gt_boxes[ix,0]) * (gt_boxes[ix,3]-gt_boxes[ix,1])
#         #print (gt_area)
#         for iy in range(num_tr):
#             flag = intersect(gt_boxes[ix], target_boxes[iy])
#             if flag:
#                 iou = calc_iou(gt_boxes[ix], target_boxes[iy])
#                 IOU_s[ix,iy] = iou
#     IOU_s = np.transpose(IOU_s)
#     return IOU_s


def calculate_IOU (target_boxes, gt_boxes):
    num_gt = gt_boxes.shape[0]
    num_tr = target_boxes.shape[0]
    IOU_s = np.zeros((num_gt,num_tr), dtype=np.float)
    for ix in range(num_gt):
        gt_area = (gt_boxes[ix,2]-gt_boxes[ix,0]) * (gt_boxes[ix,3]-gt_boxes[ix,1])
        for iy in range(num_tr):
            iw = min(gt_boxes[ix,2],target_boxes[iy,2]) - max(gt_boxes[ix,0],target_boxes[iy,0])
            if iw > 0:
                ih = min(gt_boxes[ix,3],target_boxes[iy,3]) - max(gt_boxes[ix,1],target_boxes[iy,1])
                if ih > 0:
                    tar_area = (target_boxes[iy,2]-target_boxes[iy,0]) * (target_boxes[iy,3]-target_boxes[iy,1])
                    i_area = iw * ih
                    iou = i_area/float((gt_area+tar_area-i_area))
                    IOU_s[ix,iy] = iou
    IOU_s = np.transpose(IOU_s)
    return IOU_s

def labels_generate (gt_boxes, target_boxes, overlaps_pos, overlaps_neg, im_width, im_height):
    total_targets = target_boxes.shape[0]
    targets_inside = np.where((target_boxes[:,0]>0)&\
                              (target_boxes[:,2]<im_width)&\
                              (target_boxes[:,1]>0)&\
                              (target_boxes[:,3]<im_height))[0]


    # print('total_targets : ', target_boxes.shape)
    # print('targets_inside : ', targets_inside.shape)
    targets = target_boxes[targets_inside]
    labels = np.empty((targets.shape[0],), dtype=np.float32)
    labels.fill(-1)

    IOUs = calculate_IOU(targets, gt_boxes)

    # max_gt_arg는 각 anchor 중에서 어떠한 gt object와 크게 겹치는지 (anchor마다 1개 object만 있다)
    max_gt_arg = np.argmax(IOUs, axis=1)
    max_IOUS = IOUs[np.arange(len(targets_inside)), max_gt_arg]

    # 0.3 보다 작게 겹치면 0
    labels[max_IOUS < overlaps_neg] = 0
    # 5개 object가 있을 경우, 이 object와 가장 크게 겹치는 앵커
    # (, 5) 개는 1
    max_anchor_arg = np.argmax(IOUs, axis=0)

    # print(max_anchor_arg.shape)
    labels[max_anchor_arg] = 1
    # 0.7 보다 크게 겹치는 경우 1
    labels[max_IOUS > overlaps_pos] = 1

    # bbox loss function 에서 pi는 positvie anchor에 대해서 계산
    anchor_obj = max_gt_arg
    # anchor_obj = max_anchor_arg

    # 1764개의 앵커중 조건을 만족하는 앵커 1700개에 대해서 label을 1로 채움
    # 현재 label의 크기는 1700
    labels = fill_label(labels, total_targets, targets_inside)
    # 1764개 앵커에 대해서 label을 채운 값을 return
    #set the outside anchor with label -1
    # 5개 앵커 말고는 -1로 채움
    anchor_obj = fill_label(anchor_obj, total_targets, targets_inside, fill=-1)
    anchor_obj = anchor_obj.astype(np.int32)
    return labels, anchor_obj


def labels_filt (labels, anchor_batch):
    max_fg_num = anchor_batch*0.5
    fg_inds = np.where(labels==1)[0]
    if len(fg_inds) > max_fg_num:
        disable_inds = np.random.choice(fg_inds, size=int(len(fg_inds) - max_fg_num), replace=False)
        labels[disable_inds] = -1
    max_bg_num = anchor_batch - np.sum(labels==1)
    bg_inds = np.where(labels==0)[0]
    if len(bg_inds) > max_bg_num:
        disable_inds = np.random.choice(bg_inds, size=int(len(bg_inds) - max_bg_num), replace=False)
        labels[disable_inds] = -1
    return labels

def anchor_labels_process(boxes, conners, anchor_batch, overslaps_max, overslaps_min, im_width, im_height):
    labels, anchor_obj = labels_generate(boxes, conners, overslaps_max, overslaps_min, im_width, im_height)
    # positive anchor가 128개 보다 많으면 -1로 바꿔주고
    # negaitive anchor가 (256 - positive anhcor) 보다 많으면 -1로 바꿈
    # 1 = gt, 0 = bg, -1 = None
    labels = labels_filt(labels, anchor_batch)
    return labels, anchor_obj

def fill_label(labels, total_target, target_inside, fill=-1):
    new_labels = np.empty((total_target, ), dtype=np.float32)
    new_labels.fill(fill)
    new_labels[target_inside] = labels
    return new_labels

#     print (IOUs)

if __name__ == '__main__':
    anchor_scales = [128, 256, 512]
    anchor_ratios = [0.5, 1, 2]
    all_anchor_conner(800, 272.5, anchor_scales, anchor_ratios)
    # IOUs = calculate_IOU(np.array([[233.6, 160, 441.6, 553.6]]), np.array([[222,163,402,525]]))
    # print(IOUs)