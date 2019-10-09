import os
import time
from easydict import EasyDict as edict

cfg = edict()

cfg.ROOT = 'F:\\kwon\\image-classification-master'

cfg.image_dir = 'F:\\train_object\\coco'
cfg.class_file = os.path.join(cfg.ROOT, 'data', 'classes', 'coco_classes.txt')

cfg.data_dir = os.path.join(cfg.ROOT, 'data', 'dataset', 'coco')

cfg.tfrecord_num = 100

cfg.train_num = 110000
# cfg.image_size = 224
cfg.min_image_size = 600
cfg.max_image_size = 800

cfg.summary_step = 1000
cfg.num_classes = 80 + 1 # coco + background
cfg.dropout_keep_prob = 0.5
cfg.initial_learning_rate = 0.001
cfg.decay_rate = 0.1
cfg.staircase = True
cfg.epsilon = 1e-3
cfg.decay = 0.99
cfg.weight_decay = 0.00004

cfg.batch_size = 1
cfg_num = 1281167
cfg.epochs = 100
cfg.decay_steps = 30 * (cfg_num // cfg.batch_size)


# RPN
cfg.anchor_scales = [128, 256, 512]
cfg.anchor_ratios = [0.5, 1, 2]
cfg.num_anchors= 9
cfg.max_rpn_input_num = 12000
cfg.max_nms_num = 2000
cfg.test_max_rpn_input_num = 6000
cfg.test_max_nms_num = 300
cfg.nms_thresh = 0.7

#ROI
cfg.fg_thresh = 0.5
cfg.bg_thresh_hi = 0.5
cfg.bg_thresh_lo = 0.0
cfg.test_nms_thresh = 0.3
cfg.test_fp_tp_thresh = 0.5
cfg.test_max_per_image = 10

cfg.feat_stride = 16

cfg.dect_train_batch = 128
cfg.anchor_batch = 256
cfg.overlaps_max = 0.7
cfg.overlaps_min = 0.3
cfg.POOLING_SIZE = 7

cfg.dect_fg_rate = 0.25

cfg.bbox_nor_target_pre = True
cfg.bbox_nor_mean = (0., 0., 0., 0.)
cfg.bbox_nor_stdv = (0.1, 0.1, 0.2, 0.2)

cfg.roi_input_inside_weight = (1., 1., 1., 1.)


#rpn loss
cfg.rpn_lmd = 10
cfg.rpn_sigma = 3.0
# cfg.rpn_dim2mean = 1

#head loss
cfg.roi_sigma=10
cfg.roi_dim=[1]