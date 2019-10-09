import tensorflow as tf
import numpy as np
import os
from config.config_Faster_RCNN import cfg
import cv2
from utils.faster_rcnn.anchors import *
from utils.faster_rcnn.roi import proposal_target
import time
from utils.faster_rcnn.load_coco import preprocess


# from utils.faster_rcnn.anchors_target_layer import anchor_target_layer


class Faster_RCNN:
    def __init__(self, model='Faster_RCNN', backbone='resnet101', is_train=True):
        ################
        self._proposal_targets = {}
        self._predictions = {}
        ###############
        out_dir = os.path.join(cfg.ROOT, 'logs', model)
        now = time.time()
        self.summary_dir = os.path.join(out_dir, str(now), "summaries")
        self.checkpoint_dir = os.path.join(out_dir, str(now), "checkpoints")

        self.is_train = is_train
        self.pre_train = True

        self.graph = tf.Graph()
        ConfigProto = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        ConfigProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=ConfigProto, graph=self.graph)

        # 각각의 backbone 마다 bottleneck 개수
        if backbone == 'resnet50':
            self.num_blocks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.num_blocks = [3, 4, 23, 3]
        elif backbone == 'resnet152':
            self.num_blocks = [3, 8, 36, 3]
        else:
            raise NotImplementedError

        with self.graph.as_default():

            # self.image, self.gt_boxes, self.image_height, self.image_width = iterator.get_next()

            if self.is_train:
                dataset = preprocess()
                train_data = dataset.build_dataset(cfg.batch_size)
                iterator = train_data.make_one_shot_iterator()
                # self.image, self.gt_boxes, self.gt_cls, self.image_height, self.image_width = iterator.get_next()
                self.image, self.gt_boxes, self.gt_cls, self.image_height, self.image_width, \
                self.positive_labels, self.positive_negative_labels, self.gt_positive_labels_bbox, self.gt_positive_negative_labels, self.anchors = iterator.get_next()

                self.gt_boxes = tf.squeeze(self.gt_boxes, axis=0)
                self.gt_cls = tf.squeeze(self.gt_cls, axis=0)
            else:
                self.image =  tf.placeholder(tf.float32, [1, None, None, 3])


            #################################################################################
            self.cls_score, self.cls_pred, self.cls_prob, self.bbox_pred = self.build_network()

            self.sess.run(tf.global_variables_initializer())

            save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.saver = tf.train.Saver(var_list=save_vars, max_to_keep=20)

            if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
                logs = os.listdir(out_dir)
                logs.sort()
                pre_ckpt_dir = logs.pop()
                print('out_dir : ', out_dir)
                print('pre_ckpt_dir : ', pre_ckpt_dir)
                filename = tf.train.latest_checkpoint(os.path.join(out_dir, pre_ckpt_dir, 'checkpoints'))
            else:
                filename = None

            if self.is_train:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.learning_rate = tf.train.exponential_decay(
                    cfg.initial_learning_rate, self.global_step, cfg.decay_steps,
                    cfg.decay_rate, cfg.staircase, name='learning_rate')

                self.loss_box, self.l_loss_sum = self.compute_rpn_loss()
                self.log_loss, self.reg_loss = self.compute_head_loss()
                tf.summary.scalar("rpn/class_loss", self.l_loss_sum)
                tf.summary.scalar("rpn/reg_loss", self.loss_box)
                tf.summary.scalar("head/class_loss", self.log_loss)
                tf.summary.scalar("head/reg_loss", self.reg_loss)

                self.total_loss = self.loss_box + self.l_loss_sum + self.log_loss + self.reg_loss

                self.summary_writer = tf.summary.FileWriter(self.summary_dir, graph=self.sess.graph)
                with tf.control_dependencies(save_vars):
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                                        global_step=self.global_step,
                                                                                        name='optimizer')
                # tf.summary.scalar("acc", self.accuracy)
                tf.summary.scalar('total_loss', self.total_loss)
                tf.summary.scalar("lr", self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                os.makedirs(self.summary_dir, exist_ok=True)
                os.makedirs(self.checkpoint_dir, exist_ok=True)

            self.sess.run(tf.global_variables_initializer())

            # if filename is not None:
            #     print('restore from : ', filename)
            #     self.saver.restore(self.sess, filename)
            # else:
            #     if self.pre_train:
            #         print('restore resnet.....')
            #         filename = tf.train.latest_checkpoint(
            #             'F:\\kwon\\image-classification-master\\logs\\resnet101\\1568089766.742705\\checkpoints')
            #         print(filename)
            #         restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNet")
            #         self.restore = tf.train.Saver(var_list=restore_vars)
            #         self.restore.restore(self.sess, filename)
            #     else:
            #         print('initialize....')

    def compute_head_loss(self):
        #self.cls_score, self.labels, self.bbox_pred, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights
        labels = tf.reshape(self.labels, [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_score, labels=labels))
        sigma_2 = cfg.head_sigma ** 2
        box_diff = self.bbox_pred - self.bbox_targets
        in_box_diff = self.bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = self.bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=cfg.head_dim))

        return cross_entropy, loss_box


    def compute_rpn_loss(self):
        # label은 총 1764개의 anchor 에 대해서 iou가 0.3 미만은 0 0.7 이상은, 나머지 -1
        # anchor_obj 실제 5개 object가 있는 곳만 1 이고 나머지 -1
        # 1, 0 , -1 label 중에서 1,0 만 가져옴 ( 0 = bg )
        useful_label = tf.reshape(tf.where(tf.not_equal(self.rpn_labels, -1)), [-1])
        reg_loss_nor = tf.cast(tf.shape(self.rpn_labels)[0] / 9, tf.float32)
        # 1764개 acnhor들 중에서 object가 있는것과 bg
        label_gather = tf.gather(self.rpn_labels, useful_label)
        label_gather = tf.cast(label_gather, dtype=tf.int32)
        # 1764개의 anchor들 중에서 object가 있는것과 bg
        # label_gt_order =  1764개 중 gt랑 가증 큰 anchor만 1로 되어 있는데, usfule_label 중 5개만 1이고 나머진 0 이겠지
        label_gt_order = tf.gather(self.anchor_obj, useful_label)
        # 전체 acnhor 중에서 object가 있는것과 bg
        anchor = tf.gather(self.anchors, useful_label)

        # self.probability = self.cls
        # self.probability = tf.squeeze(self.probability)
        # self.probability = tf.reshape(self.probability, [-1, 9 * 2])
        # 모델이 object가 있다고 예측한 것과 gt를 비교
        # [batch, 14, 14 , 18] 에서 1번째 idx가 object가 있다
        # 14 * 14 * 9 = 1764
        self.probability = tf.reshape(self.cls, [-1, 2])
        self.probability_gather = tf.gather(self.probability, useful_label)
        self.probability_gather = tf.cast(self.probability_gather, dtype=tf.float32)

        # gather the prediction_bbox to be computed in reg_loss
        # self.prediction_bbox = tf.squeeze(self.prediction_bbox)
        # self.prediction_bbox = tf.reshape(self.prediction_bbox, [-1, 9 * 4])
        self.prediction_bbox = tf.reshape(self.bbox, [-1, 4])
        self.prediction_bbox_gather = tf.gather(self.prediction_bbox, useful_label)

        # reconsitution_coords
        # anchor_x1 = anchor[:, 0]
        # anchor_y1 = anchor[:, 1]
        # anchor_x2 = anchor[:, 2]
        # anchor_y2 = anchor[:, 3]
        anchor_x1 = anchor[:, 0]
        anchor_y1 = anchor[:, 1]
        anchor_x2 = anchor[:, 2]
        anchor_y2 = anchor[:, 3]


        re_anchor_0 = tf.cast((anchor_x2 + anchor_x1) / 2.0, dtype=tf.float32)
        re_anchor_1 = tf.cast((anchor_y2 + anchor_y1) / 2.0, dtype=tf.float32)
        re_anchor_2 = tf.cast((anchor_x2 - anchor_x1), dtype=tf.float32)
        re_anchor_3 = tf.cast((anchor_y2 - anchor_y1), dtype=tf.float32)
        re_anchor = tf.squeeze(tf.stack(
            [re_anchor_0, re_anchor_1, re_anchor_2, re_anchor_3], axis=1))

        ground_truth_x1 = self.gt_boxes[:, 0]
        ground_truth_y1 = self.gt_boxes[:, 1]
        ground_truth_x2 = self.gt_boxes[:, 2]
        ground_truth_y2 = self.gt_boxes[:, 3]

        re_ground_truth_0 = tf.expand_dims(tf.cast((ground_truth_x1 + ground_truth_x2) / 2.0, dtype=tf.float32), -1)
        re_ground_truth_1 = tf.expand_dims(tf.cast((ground_truth_y1 + ground_truth_y2) / 2.0, dtype=tf.float32), -1)
        re_ground_truth_2 = tf.expand_dims(tf.cast((ground_truth_x2 - ground_truth_x1 + 1.0), dtype=tf.float32), -1)
        re_ground_truth_3 = tf.expand_dims(tf.cast((ground_truth_y2 - ground_truth_y1 + 1.0), dtype=tf.float32), -1)
        re_ground_truth = tf.concat([re_ground_truth_0, re_ground_truth_1, re_ground_truth_2, re_ground_truth_3],
                                         axis=1)

        # self.gt_map=tf.one_hot(self.label_gt_order,self.size)
        # self.re_label_gt_order=tf.matmul(self.gt_map,self.re_ground_truth)
        # self.re_label_gt_order=tf.cast(self.re_label_gt_order,dtype=tf.float32)

        # 실제 object가 일정 이상 겹친 acnhor들 = label_gt_order
        # re_ground_truth는 실제 object에 대해서 x,y,w,h
        #TODO
        # object가 5개 일때 re_ground_truth 크기는 5개, label_gt_order 는 좀 클 텐데???
        self.re_label_gt_order = tf.gather(re_ground_truth, label_gt_order)

        # chosse which rpn_box to be computed in reg_loss, for label is positive ie, 1
        # label_gather는 1764개중 label이 backround나 존재하는 것들
        #  label_weight_c는 object라 0.7 이상 겹치는 애들
        self.label_weight_c = tf.cast((label_gather > 0), tf.float32)
        # label_weight_c는 bg가 아닌 실제 object가 있는 것들
        self.label_weight_c = tf.expand_dims(self.label_weight_c, axis=1)

        # object가  있는지 없는지에 대한 loss
        l_loss_sum = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.probability_gather, labels=label_gather))

        # bbox_predicted = prediction_bbox_gather
        # bbox_ground_truth = self.re_label_gt_order
        # weight = self.label_weight_c

        sigma_1 = cfg.rpn_sigma ** 2
        bbox_ground_truth_0 = tf.cast((self.re_label_gt_order[:, 0] - re_anchor_0) / re_anchor_2, dtype=tf.float32)
        bbox_ground_truth_1 = tf.cast((self.re_label_gt_order[:, 1] - re_anchor_1) / re_anchor_3, dtype=tf.float32)
        bbox_ground_truth_2 = tf.cast(tf.log(self.re_label_gt_order[:, 2] / re_anchor_2), dtype=tf.float32)
        bbox_ground_truth_3 = tf.cast(tf.log(self.re_label_gt_order[:, 3] / re_anchor_3), dtype=tf.float32)
        re_bbox_ground_truth = tf.stack(
            [bbox_ground_truth_0, bbox_ground_truth_1, bbox_ground_truth_2, bbox_ground_truth_3], axis=1)
        # re_bbox_predicted = bbox_predicted
        bbox_diff = self.prediction_bbox_gather - re_bbox_ground_truth
        t_diff = bbox_diff * self.label_weight_c
        t_diff_abs = tf.abs(t_diff)
        compare_1 = tf.stop_gradient(tf.to_float(tf.less(t_diff_abs, 1.0 / sigma_1)))
        # compare_1 = tf.to_float(tf.less(t_diff_abs, 1.0/sigma_1))
        sl_loss_box = (sigma_1 / 2.0) * compare_1 * tf.pow(t_diff_abs, 2) + (1.0 - compare_1) * (
        t_diff_abs - 0.5 / sigma_1)
        sum_loss_box = tf.reduce_sum(sl_loss_box)
        loss_box = sum_loss_box * cfg.rpn_lmd / cfg.anchor_batch

        return loss_box, l_loss_sum


    def build_network(self):
        with tf.variable_scope('ResNet'):
            model = tf.layers.conv2d(inputs=self.image, filters=64, kernel_size=(7, 7), strides=(2, 2),
                                     padding='SAME', name='conv_1',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay))
            model = tf.layers.batch_normalization(inputs=model, trainable=True, training=self.is_train)
            model = tf.nn.relu(model)
            model = tf.layers.max_pooling2d(model, pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pooling')
            print(model)

            with tf.variable_scope('layers_2n'):
                for idx in range(self.num_blocks[0]):
                    block_name = 'layers_2n_{}'.format(idx)
                    model = self.residual_bottleneck(block_name, model, filters=64, kernel_size=3, stride=1)
                    print(block_name, model)
            with tf.variable_scope('layers_4n'):
                for idx in range(self.num_blocks[1]):
                    block_name = 'layers_4n_{}'.format(idx)
                    # 첫번째 block에서 down sampling
                    if idx == 0:
                        model = self.residual_bottleneck(block_name, model, filters=128, kernel_size=3, stride=2)
                    else:
                        model = self.residual_bottleneck(block_name, model, filters=128, kernel_size=3, stride=1)
                    print(block_name, model)
            with tf.variable_scope('layers_6n'):
                for idx in range(self.num_blocks[2]):
                    block_name = 'layers_6n_{}'.format(idx)
                    if idx == 0:
                        model = self.residual_bottleneck(block_name, model, filters=256, kernel_size=3, stride=2)
                    else:
                        model = self.residual_bottleneck(block_name, model, filters=256, kernel_size=3, stride=1)
                    print(block_name, model)
            # with tf.variable_scope('layers_8n'):
            #     for idx in range(self.num_blocks[3]):
            #         block_name = 'layers_8n_{}'.format(idx)
            #         if idx == 0:
            #             model = self.residual_bottleneck(block_name, model, filters=512, kernel_size=3, stride=2)
            #         else:
            #             model = self.residual_bottleneck(block_name, model, filters=512, kernel_size=3, stride=1)
            #         print(block_name, model)


        with tf.variable_scope('RPN'):
            # 14 * 14* 1024
            rpn = tf.layers.conv2d(inputs=model, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                   name='intermediate', activation=tf.nn.relu)
            print(model)
            with tf.variable_scope('cls'):
                self.cls = tf.layers.conv2d(inputs=rpn, filters=cfg.num_anchors * 2, kernel_size=(1, 1), strides=(1, 1),
                                       padding='SAME', activation=None)
            with tf.variable_scope('bbox'):
                self.bbox = tf.layers.conv2d(inputs=rpn, filters=cfg.num_anchors * 4, kernel_size=(1, 1), strides=(1, 1),
                                        padding='SAME', activation=None)
            print('cls : ', self.cls)
            print('bbox : ', self.bbox)
            # self.anchors = tf.py_func(all_anchor_conner, [self.image_width, self.image_height, cfg.feat_stride],
            self.anchors = tf.py_func(all_anchor_conner, [self.image_width, self.image_height, cfg.anchor_scales, cfg.anchor_ratios, cfg.feat_stride],
                                      tf.float32)
            # self.labels, self.anchor_obj = anchor_labels_process(self.gt_boxes, self.anchors, cfg.anchor_batch,  cfg.overlaps_max, cfg.overlaps_min, self.image_width, self.image_height)

            if self.is_train:
                self.rpn_labels, self.anchor_obj = tf.py_func(anchor_labels_process,
                                                          [self.gt_boxes, self.anchors, cfg.anchor_batch,
                                                           cfg.overlaps_max, cfg.overlaps_min, self.image_width,
                                                           self.image_height], [tf.float32, tf.int32])
                print('self.labels : ', self.rpn_labels)
                print('self.anchor_obj : ', self.anchor_obj)
            # [batch, 7 ,7, 2*9] => [batch * 7 * 7, 2*9]
            # bbox = tf.squeeze(bbox)
            # bbox = tf.reshape(bbox, [-1, cfg.num_anchors * 4])
            bbox = tf.reshape(self.bbox, [-1, 4])

            # cls = tf.squeeze(cls)
            # cls = tf.reshape(cls, [-1, cfg.num_anchors * 2])
            cls = tf.reshape(self.cls, [-1, 2])
            print('cls : ', cls)
            print('bbox : ', bbox)

        with tf.variable_scope('ROI'):
            if self.is_train:
                post_nms_topN = cfg.max_nms_num
            else:
                post_nms_topN = cfg.test_max_nms_num

            nms_thresh = cfg.nms_thresh
            # idx 0 : object가 없다
            # idx 1 : object가 있다
            scores = cls[:, 1]
            rpn_bbox_pred = bbox

            # anchor는 feature map ^ 2 * num anchors, (x1, y1, x2, y2)
            # anchors_x = x1 + x2 / 2
            # anchors_y = y1 + y2 / 2

            # all_anchor_conners: (196 * 9, 4)
            print(self.anchors)
            anchor_x = tf.add(self.anchors[:, 2], self.anchors[:, 0]) * 0.5
            anchor_y = tf.add(self.anchors[:, 3], self.anchors[:, 1]) * 0.5
            acnhor_w = tf.subtract(self.anchors[:, 2], self.anchors[:, 0]) + 1.0
            acnhor_h = tf.subtract(self.anchors[:, 3], self.anchors[:, 1]) + 1.0

            # 기존 앵커 값들은 다 정해져있으니, model이 내뱉는 값을에 acnhor 값을 곱해줌
            # 모델이 각 anchor마다 예측하는 4개의 좌표가 나옴
            # cood 값은 gt bbox 처럼 이미지 전체에서 좌표 값들임 (open cv2가 rectangle 그리듯이)
            bbox_x = bbox[:, 0] * acnhor_w + anchor_x
            bbox_y = bbox[:, 1] * acnhor_h + anchor_y
            bbox_w = tf.exp(bbox[:, 2]) * acnhor_w
            bbox_h = tf.exp(bbox[:, 3]) * acnhor_h

            # model이 예측한 bbox으 좌표
            coord_x1 = bbox_x - bbox_w * 0.5
            coord_y1 = bbox_y - bbox_h * 0.5
            coord_x2 = bbox_x + bbox_w * 0.5
            coord_y2 = bbox_y + bbox_h * 0.5
            coord_result = tf.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)
            print('coord_result : ', coord_result)
            # coord result는 model이 예측한 값을 anchor에 맞게 값을 변환 한 값임
            # 원본 이미지에서 각각의 앵커에 대해서 예측한 좌표값들

            # 좌표에서 min max 보정
            b0 = tf.maximum(tf.minimum(coord_result[:, 0], tf.cast((self.image_width - 1), tf.float32)), 0.0)
            b1 = tf.maximum(tf.minimum(coord_result[:, 1], tf.cast((self.image_height - 1), tf.float32)), 0.0)
            b2 = tf.maximum(tf.minimum(coord_result[:, 2], tf.cast((self.image_width - 1), tf.float32)), 0.0)
            b3 = tf.maximum(tf.minimum(coord_result[:, 3], tf.cast((self.image_height - 1), tf.float32)), 0.0)
            coord_result = tf.stack([b0, b1, b2, b3], axis=1)
            print('coord_result : ', coord_result)
            print('scores : ', scores)

            # 1764 개에 대해서 NMS 해줌
            # [1764, 4]
            # [1764, 1]
            inds = tf.image.non_max_suppression(coord_result, scores, max_output_size=post_nms_topN,
                                                iou_threshold=nms_thresh)
            # 1764, 1]
            boxes = tf.gather(coord_result, inds)
            boxes = tf.to_float(boxes)
            roi_scores = tf.gather(scores, inds)
            # roi_scores = tf.reshape(scores, shape=(-1, 1))
            batch_inds = tf.zeros((tf.shape(inds)[0], 1), dtype=tf.float32)
            # 모델이 예측한 좌표값에 대해서 NMS 한 결과 = RoI
            rois = tf.concat([batch_inds, boxes], 1)
            print('roi_scores : ', roi_scores)
            print('batch_inds : ', batch_inds)
            print('rois : ', rois)
            # rois_coord = rois
            # rios_scroe_process = roi_scores

            # 학습 할 때는 target layer에 대해서 proposal
            # RoI 중에서 256 batch 에 대해서 positive와 negative sample을 만듦
            if self.is_train:
                with tf.variable_scope('process'):
                    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                        proposal_target,
                        [rois, roi_scores, self.gt_boxes, cfg.num_classes, self.gt_cls],
                        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                        name="proposal_target")
                rois.set_shape([cfg.dect_train_batch, 5])
                roi_scores.set_shape([cfg.dect_train_batch])
                labels.set_shape([cfg.dect_train_batch, 1])
                bbox_targets.set_shape([cfg.dect_train_batch, cfg.num_classes * 4])
                bbox_inside_weights.set_shape([cfg.dect_train_batch, cfg.num_classes * 4])
                bbox_outside_weights.set_shape([cfg.dect_train_batch, cfg.num_classes * 4])

                # self._proposal_targets['rois'] = rois
                # self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
                # self._proposal_targets['bbox_targets'] = bbox_targets
                # self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
                # self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights,

                self.labels = tf.to_int32(labels, name="to_int32")
                self.bbox_targets = bbox_targets
                self.bbox_inside_weights = bbox_inside_weights
                self.bbox_outside_weights = bbox_outside_weights
            self.rois = rois

        ########################################
        # train 에서는 256개 대해서
        # infernce에서는 전체 roi 대해서
        with tf.variable_scope('roi_pooing'):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            print('batch_ids : ', batch_ids)
            bottom_shape = tf.shape(model)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(cfg.feat_stride)
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(cfg.feat_stride)
            # RoI는 원본이미지에서 모델이 예측한 좌표 값들임
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2  # 7*2
            print('bboxes : ', bboxes)
            print('model : ', model)
            print('pre_pool_size : ', pre_pool_size)
            print('batch_ids : ', batch_ids)

            # http://incredible.ai/deep-learning/2018/03/17/Faster-R-CNN/
            # Fixed-size Resize instead of ROI Pooling
            crops = tf.image.crop_and_resize(model, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], method="bilinear",
                                             name="crops")
            print('crops : ', crops)
            crops = tf.layers.max_pooling2d(crops, pool_size=(2, 2), strides=(2, 2), padding='VALID')
            print('crops : ', crops)

        with tf.variable_scope('head'):
            crops = tf.layers.flatten(crops)
            model = tf.layers.dense(crops, 2048)
            print(model)

        with tf.variable_scope('classification'):
            cls_score = tf.layers.dense(model, cfg.num_classes,
                                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay),
                                        activation=None, name='cls_score')

            cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

            bbox_pred = tf.layers.dense(model, cfg.num_classes * 4,
                                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay),
                                        activation=None, name='bbox_pred')

            # self._predictions["cls_score"] = cls_score
            # self._predictions["cls_pred"] = cls_pred
            # self._predictions["cls_prob"] = cls_prob
            # self._predictions["bbox_pred"] = bbox_pred
        return cls_score, cls_pred, cls_prob, bbox_pred

    def residual_bottleneck(self, name, inputs, kernel_size, filters, stride):
        # orginal resnet은 stride가 첫 번째 convolution
        # 추후에는 정보량 손실로 인해 두 번재 convolution에 stride
        with tf.variable_scope(name):
            # 1x1
            block = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1, 1), strides=(1, 1),
                                     padding='SAME',
                                     name='conv_1', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay))
            block = tf.layers.batch_normalization(inputs=block, trainable=True, training=self.is_train)
            block = tf.nn.relu(block)

            # 3x3
            block = tf.layers.conv2d(inputs=block, filters=filters, kernel_size=(kernel_size, kernel_size),
                                     strides=(stride, stride),
                                     padding='SAME',
                                     name='conv_2', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay))
            block = tf.layers.batch_normalization(inputs=block, trainable=True, training=self.is_train)
            block = tf.nn.relu(block)

            # 1*1
            block = tf.layers.conv2d(inputs=block, filters=filters * 4, kernel_size=(1, 1), strides=(1, 1),
                                     padding='SAME',
                                     name='conv_3', kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay))
            block = tf.layers.batch_normalization(inputs=block, trainable=True, training=self.is_train)
            block = tf.nn.relu(block)

            # shortcut
            if int(block.shape[3]) != int(inputs.shape[3]):
                # inputs = tf.pad(inputs, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_1')
                inputs = tf.layers.conv2d(inputs, filters=filters * 4, kernel_size=(1, 1),
                                          strides=stride, padding='VALID')

            return tf.nn.relu(block + inputs)

    def train(self, save=False):
        with self.graph.as_default():
            # feed_dict = {self.input_x: batch_x,
            #              self.label: batch_y}
            _, global_step, summary_str, loss, lr = self.sess.run(
                [self.train_op, self.global_step, self.summary_op, self.total_loss, self.learning_rate],
            )
            if save:
                self.summary_writer.add_summary(summary_str, global_step=global_step)
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            return global_step, loss, acc, lr


if __name__ == '__main__':
    net = Faster_RCNN('Faster_RCNN', 'resnet101', True)

    for epoch in range(cfg.epochs):
        loss = 0
        acc = 0
        for step in range(int(cfg.train_num / cfg.batch_size)):
            start_time = time.time()
            if step % 200 == 0 and step != 0:
                global_step, train_loss, train_acc, lr = net.train(True)
            else:
                global_step, train_loss, train_acc, lr = net.train(False)
            end_time = time.time()
            print('Epoch {} step {}, loss = {}, acc = {} , processing time = {} lr = {}'.format(epoch, global_step,
                                                                                                train_loss, train_acc,
                                                                                                end_time - start_time,
                                                                                                lr))
            loss += train_loss
            acc += train_acc

        print('Epoch {} step {}, loss = {}, acc = {}'.format(epoch, global_step, loss / step, acc / step))
