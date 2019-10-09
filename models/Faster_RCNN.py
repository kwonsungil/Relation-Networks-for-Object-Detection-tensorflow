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
        # self.pre_train = True
        self.pre_train = False

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

            if self.is_train:
                dataset = preprocess()
                train_data = dataset.build_dataset(cfg.batch_size)
                iterator = train_data.make_one_shot_iterator()

                self.image, self.gt_boxes, self.gt_cls, self.image_height, self.image_width , \
                self.positive_labels, self.positive_negative_labels, self.gt_positive_labels_bbox, self.gt_positive_negative_labels, self.anchors = iterator.get_next()

                self.anchors = tf.squeeze(self.anchors, axis=0)
                # self.image, self.gt_boxes, self.gt_cls, self.image_height, self.image_width = iterator.get_next()

                self.gt_boxes = tf.squeeze(self.gt_boxes, axis=0)
                self.gt_cls = tf.squeeze(self.gt_cls, axis=0)
            else:
                self.image =  tf.placeholder(tf.float32, [1, None, None, 3])
                self.image_height =  tf.placeholder(tf.float32, [None])
                self.image_width =  tf.placeholder(tf.float32, [None])
                self.anchors = tf.placeholder(tf.float32, [None, 4])


            #################################################################################
            model = self.build_backbone_network()
            self.cls, self.bbox = self.build_rpn_network(model)
            # self.cls_score, self.cls_pred, self.cls_prob, self.bbox_pred = self.build_roi_network(model)
            # self.cls_score, self.bbox_score, self.nms_idx = self.build_roi_network(model)
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

                self.rpn_box_loss, self.rpn_cls_loss = self.compute_rpn_loss()
                tf.summary.scalar("rpn/cls_loss", self.rpn_cls_loss)
                tf.summary.scalar("rpn/box_loss", self.rpn_box_loss)
                self.total_loss = self.rpn_box_loss + self.rpn_cls_loss

                # self.roi_cls_loss, self.roi_box_loss = self.compute_roi_loss()
                # tf.summary.scalar("roi/class_loss", self.roi_cls_loss)
                # tf.summary.scalar("roi/reg_loss", self.roi_box_loss)

                # self.total_loss = self.loss_box + self.loss_cls + self.log_loss + self.reg_loss

                self.summary_writer = tf.summary.FileWriter(self.summary_dir, graph=self.sess.graph)
                with tf.control_dependencies(save_vars):
                    # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                    self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                                        global_step=self.global_step,
                                                                                        name='optimizer')
                # tf.summary.scalar("acc", self.accuracy)
                tf.summary.scalar('total_loss', self.total_loss)
                tf.summary.scalar("lr", self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                os.makedirs(self.summary_dir, exist_ok=True)
                os.makedirs(self.checkpoint_dir, exist_ok=True)

            self.sess.run(tf.global_variables_initializer())

            if filename is not None:
                print('restore from : ', filename)
                # self.saver.restore(self.sess, filename)
                restore_rpn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="RPN")
                restore = tf.train.Saver(var_list=restore_rpn_vars)
                restore.restore(self.sess, filename)
                restore_resnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNet")
                restore = tf.train.Saver(var_list=restore_resnet_vars)
                restore.restore(self.sess, filename)
            else:
                if self.pre_train:
                    print('restore resnet.....')
                    filename = tf.train.latest_checkpoint(
                        'F:\\kwon\\image-classification-master\\logs\\resnet101\\1568185001.6838155\\checkpoints')
                    print(filename)
                    restore_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNet")
                    restore = tf.train.Saver(var_list=restore_vars)
                    restore.restore(self.sess, filename)
                else:
                    print('initialize....')

    def compute_roi_loss(self):
        labels = tf.reshape(self.labels, [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_score, labels=labels))
        sigma_1 = cfg.roi_sigma ** 2
        box_diff = self.bbox_score - self.bbox_targets
        in_box_diff = self.bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_1)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_1 / 2.) * smoothL1_sign  + (abs_in_box_diff - (0.5 / sigma_1)) * (1. - smoothL1_sign)
        out_loss_box = self.bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=cfg.roi_dim)) * 10

        return cross_entropy, loss_box

    def compute_rpn_loss(self):
        # self.probability = tf.reshape(self.cls, [-1, 2])
        positive_negative_cls = tf.gather(self.cls, self.positive_negative_labels)
        positive_negative_cls = tf.cast(positive_negative_cls, dtype=tf.float32)

        # object가  있는지 없는지에 대한 loss
        rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=positive_negative_cls, labels=self.gt_positive_negative_labels))

        prediction_bbox_gather = tf.gather(self.bbox, self.positive_labels)
        bbox_diff = prediction_bbox_gather - self.gt_positive_labels_bbox
        bbox_diff = tf.abs(bbox_diff)

        sigma_1 = cfg.rpn_sigma ** 2
        # smooth = tf.stop_gradient(tf.to_float(tf.less(bbox_diff, 1.0 / sigma_1)))
        # smooth_loss_box = (sigma_1 / 2.0) * smooth * tf.pow(bbox_diff, 2) + (1.0 - smooth) * (bbox_diff - (0.5 / sigma_1))

        # |x| < 1, 0.5x^2
        # otherwise [x] - 0.5
        # flat_bbox_diff = tf.reshape(bbox_diff, [-1])
        # bbox_diff_less= tf.boolean_mask(flat_bbox_diff, tf.less(flat_bbox_diff, 1.0 / sigma_1))
        # bbox_diff_more = tf.boolean_mask(flat_bbox_diff, tf.greater_equal(flat_bbox_diff, 1.0 / sigma_1))
        # smooth_loss_box_1 = sigma_1 / 2.0 * tf.pow(bbox_diff_less, 2)
        # smooth_loss_box_2 = (bbox_diff_more - (0.5 / sigma_1))
        # smooth_loss_box = tf.concat([smooth_loss_box_1, smooth_loss_box_2], axis=0)

        smooth_loss_box = tf.where(tf.less(bbox_diff, 1/sigma_1), 0.5 * tf.pow(bbox_diff, 2) * sigma_1, bbox_diff - 0.5 / sigma_1)
        rpn_box_loss = tf.reduce_sum(smooth_loss_box)
        rpn_box_loss = rpn_box_loss / tf.cast(tf.shape(self.positive_negative_labels)[1], tf.float32) * cfg.rpn_lmd

        return rpn_box_loss, rpn_cls_loss


    def build_backbone_network(self):
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

            return model

    def build_rpn_network(self, model):
        with tf.variable_scope('RPN'):
            # 14 * 14* 1024
            rpn = tf.layers.conv2d(inputs=model, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                                   name='intermediate', activation=tf.nn.relu)
            # with tf.variable_scope('cls'):
            cls = tf.layers.conv2d(inputs=rpn, filters=cfg.num_anchors * 2, kernel_size=(1, 1), strides=(1, 1),
                                       padding='SAME', activation=None)
            # with tf.variable_scope('bbox'):
            bbox = tf.layers.conv2d(inputs=rpn, filters=cfg.num_anchors * 4, kernel_size=(1, 1), strides=(1, 1),
                                        padding='SAME', activation=None)

            bbox = tf.squeeze(bbox)
            bbox = tf.reshape(bbox, [-1, cfg.num_anchors * 4])
            bbox = tf.reshape(bbox, [-1, 4])

            cls = tf.squeeze(cls)
            cls = tf.reshape(cls, [-1, cfg.num_anchors * 2])
            cls = tf.reshape(cls, [-1, 2])
            return cls, bbox

    def build_roi_network(self, model):
        ########################################
        #Region of Interest
        ########################################
        with tf.variable_scope('RoI'):
            if self.is_train:
                nms_num = cfg.max_nms_num
            else:
                nms_num = cfg.test_max_nms_num

            nms_thresh = cfg.nms_thresh
            # idx 0 : object가 없다
            # idx 1 : object가 있다
            cls = tf.nn.softmax(self.cls)
            scores = cls[:, 1]

            # anchor x1,y1,x2,y2 => x,y,w,h
            anchor_x = tf.add(self.anchors[:, 2], self.anchors[:, 0]) * 0.5
            anchor_y = tf.add(self.anchors[:, 3], self.anchors[:, 1]) * 0.5
            acnhor_w = tf.subtract(self.anchors[:, 2], self.anchors[:, 0]) + 1.0
            acnhor_h = tf.subtract(self.anchors[:, 3], self.anchors[:, 1]) + 1.0

            # 기존 앵커 값들은 다 정해져있으니, model이 내뱉는 값을에 acnhor 값을 곱해줌
            # 모델이 각 anchor마다 예측하는 4개의 좌표가 나옴
            # cood 값은 gt bbox 처럼 이미지 전체에서 좌표 값들임 (open cv2가 rectangle 그리듯이)
            # model이 예측한 bbox의 좌표(x, y, w, h)

            prdict_x = self.bbox[:, 0] * acnhor_w + anchor_x
            prdict_y = self.bbox[:, 1] * acnhor_h + anchor_y
            prdict_w = tf.exp(self.bbox[:, 2]) * acnhor_w
            prdict_h = tf.exp(self.bbox[:, 3]) * acnhor_h

            # model이 예측한 bbox의 좌표(x1, y1, x2, y2)
            # nms need x1,y1,x2,y2 instead of x,y,w,h
            predcit_x1 = prdict_x - prdict_w * 0.5
            predcit_y1 = prdict_y - prdict_h * 0.5
            predcit_x2 = prdict_x + prdict_w * 0.5
            predcit_y2 = prdict_y + prdict_h * 0.5
            predict_coord = tf.stack([predcit_x1, predcit_y1, predcit_x2, predcit_y2], axis=1)
            # predcit result는 model이 예측한 값을 anchor에 맞게 값을 변환 한 값임
            # 원본 이미지에서 각각의 앵커에 대해서 예측한 좌표값들

            # 좌표에서 min max 보정
            predcit_x1_ = tf.maximum(tf.minimum(predict_coord[:, 0], tf.cast((self.image_width - 1), tf.float32)), 0.0)
            predcit_y1_ = tf.maximum(tf.minimum(predict_coord[:, 1], tf.cast((self.image_height - 1), tf.float32)), 0.0)
            predcit_x2_ = tf.maximum(tf.minimum(predict_coord[:, 2], tf.cast((self.image_width - 1), tf.float32)), 0.0)
            predcit_y2_ = tf.maximum(tf.minimum(predict_coord[:, 3], tf.cast((self.image_height - 1), tf.float32)), 0.0)
            predict_coord = tf.stack([predcit_x1_, predcit_y1_, predcit_x2_, predcit_y2_], axis=1)

            nms_idx = tf.image.non_max_suppression(predict_coord, scores, max_output_size=nms_num, iou_threshold=nms_thresh)
            rois = tf.gather(predict_coord, nms_idx)
            rois_score = tf.gather(scores, nms_idx)
            # self.nms_idx = nms_idx
            # self.nms_idx = tf.reshape(nms_idx, [tf.shape(rois)[0]])
            # self.nms_idx = tf.reshape(self.nms_idx, [-1])
            # 모델이 예측한 좌표값에 대해서 NMS 한 결과
            # rois = tf.concat([batch_idxs, nms_predict_coord], 1)
            print('rois_score : ', rois_score)
            print('batch_inds : ', nms_idx)
            print('rois : ', rois)

            # 학습 할 때는 target layer에 대해서 proposal
            # RoI 중에서 256 batch 에 대해서 positive와 negative sample을 만듦
            if self.is_train:
                rois, rois_score, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                        proposal_target,
                        [rois, rois_score, self.gt_boxes, cfg.num_classes, self.gt_cls],
                        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                        name="proposal_target")
                # rois.set_shape([cfg.dect_train_batch, 5])
                rois.set_shape([cfg.dect_train_batch, 4])
                rois_score.set_shape([cfg.dect_train_batch])
                labels.set_shape([cfg.dect_train_batch, 1])
                bbox_targets.set_shape([cfg.dect_train_batch, cfg.num_classes * 4])
                bbox_inside_weights.set_shape([cfg.dect_train_batch, cfg.num_classes * 4])
                bbox_outside_weights.set_shape([cfg.dect_train_batch, cfg.num_classes * 4])

                self.labels = tf.to_int32(labels)
                self.bbox_targets = bbox_targets
                self.bbox_inside_weights = bbox_inside_weights
                self.bbox_outside_weights = bbox_outside_weights

        ########################################
        # RoI Poolling
        ########################################
        # train 에서는 256개 대해서
        # infernce에서는 NMS roi 대해서
        with tf.variable_scope('RoI_pooing'):
            # batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1]), [1])
            # bottom_shape = tf.shape(model)
            # height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(cfg.feat_stride)
            # width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(cfg.feat_stride)
            # RoI는 원본이미지에서 모델이 예측한 좌표 값들임
            # x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            # y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            # x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            # y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            x1, y1, x2, y2 = tf.split(value=rois, num_or_size_splits=4, axis=1)
            x1 = x1 / self.image_width
            y1 = y1 / self.image_height
            x2 = x2 / self.image_width
            y2 = y2 / self.image_height
            rois = tf.concat([x1, y1, x2, y2], 1)

            # Won't be back-propagated to rois anyway, but to save time
            # bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.POOLING_SIZE * 2  # 7*2

            print('rois : ', rois)
            print('model : ', model)
            box_ind = tf.zeros((tf.shape(rois)[0]), dtype=tf.float32)

            # http://incredible.ai/deep-learning/2018/03/17/Faster-R-CNN/
            # Fixed-size Resize instead of ROI Pooling
            crops = tf.image.crop_and_resize(model, rois, tf.to_int32(box_ind), [pre_pool_size, pre_pool_size], method="bilinear",
                                             name="crops")
            crops = tf.layers.max_pooling2d(crops, pool_size=(2, 2), strides=(2, 2), padding='VALID')

            crops = tf.layers.flatten(crops)

            model = tf.layers.dense(crops, 4096,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                    activation=tf.nn.relu)
            model = tf.layers.dense(model, 4096,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                    activation=tf.nn.relu)
            cls_score = tf.layers.dense(model, cfg.num_classes,
                                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                        # kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay),
                                        activation=None, name='cls_score')

            # cls_prob = tf.nn.softmax(cls_score, name="cls_prob")
            # cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

            bbox_score = tf.layers.dense(model, cfg.num_classes * 4,
                                        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                        # kernel_regularizer=tf.contrib.layers.l2_regularizer(cfg.weight_decay),
                                        activation=None, name='bbox_pred')

        # return cls_score, cls_pred, cls_prob, bbox_pred
        return cls_score, bbox_score, nms_idx

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

            _, global_step, summary_str, lr, roi_cls_loss, roi_box_loss, rpn_box_loss, rpn_cls_loss, gt_positive_labels_bbox, positive_labels, bbox_score = self.sess.run(
                [self.train_op, self.global_step, self.summary_op, self.learning_rate,
                 self.roi_cls_loss, self.roi_box_loss, self.rpn_box_loss, self.rpn_cls_loss,
                 # self.prediction_bbox_gather, self.gt_positive_labels_bbox, self.positive_labels],
                 self.gt_positive_labels_bbox, self.positive_labels, self.bbox_score],
            )
            print('roi_cls_loss, roi_box_loss, rpn_box_loss, rpn_cls_loss : ', roi_cls_loss, roi_box_loss, rpn_box_loss, rpn_cls_loss)
            print('bbox_score : ', bbox_score)
            # print('sl_loss_box : ', sl_loss_box)
            # print('smooth : ', smooth)
            # print('prediction_bbox_gather : ', prediction_bbox_gather)
            # print('gt_positive_labels_bbox : ', gt_positive_labels_bbox)
            # print('positive_labels : ', positive_labels)

            if save:
                self.summary_writer.add_summary(summary_str, global_step=global_step)
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            return global_step, loss, acc, lr

    def train_rpn(self, save=False):
        with self.graph.as_default():
            # smooth_loss_box_1, smooth_loss_box_2, positive_labels = self.sess.run([self.smooth_loss_box_1, self.smooth_loss_box_2, self.positive_labels])
            # print(smooth_loss_box_1.shape)
            # print(smooth_loss_box_2.shape)
            # print(positive_labels.shape)
            _, global_step, summary_str, lr, rpn_box_loss, rpn_cls_loss, total_loss, gt_positive_labels_bbox = self.sess.run(
            # _, global_step, summary_str, lr, prediction_bbox_gather, gt_positive_labels_bbox, positive_labels = self.sess.run(
                [self.train_op, self.global_step, self.summary_op, self.learning_rate,
                 self.rpn_box_loss, self.rpn_cls_loss, self.total_loss, self.gt_positive_labels_bbox],
            )
            if save:
                self.summary_writer.add_summary(summary_str, global_step=global_step)
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            print('rpn_box_loss, rpn_cls_loss : ', rpn_box_loss, rpn_cls_loss)
            # print('gt_positive_labels_bbox : ', gt_positive_labels_bbox)

            return global_step, total_loss, acc, lr


if __name__ == '__main__':
    net = Faster_RCNN('Faster_RCNN', 'resnet101', True)

    for epoch in range(cfg.epochs):
        loss = 0
        acc = 0
        for step in range(int(cfg.train_num / cfg.batch_size)):
            start_time = time.time()
            if step % 5000 == 0 and step != 0:
                # global_step, train_loss, train_acc, lr = net.train(True)
                global_step, train_loss, train_acc, lr = net.train_rpn(True)
            else:
                # net.train(False)
                # global_step, train_loss, train_acc, lr = net.train(False)
                global_step, train_loss, train_acc, lr = net.train_rpn(False)
            end_time = time.time()
            print('Epoch {} step {}, loss = {}, acc = {} , processing time = {} lr = {}'.format(epoch, global_step,
                                                                                                train_loss, train_acc,
                                                                                                end_time - start_time,
                                                                                                lr))
            loss += train_loss
            acc += train_acc

        print('Epoch {} step {}, loss = {}, acc = {}'.format(epoch, global_step, loss / step, acc / step))
