import os
import tensorflow as tf
from config.config_Faster_RCNN import cfg
import random
import numpy as np
import skimage.io
from utils.faster_rcnn.anchors import all_anchor_conner, anchor_labels_process
import cv2

import xml.etree.ElementTree as ET

class preprocess:
    def __init__(self, is_train=True):
        # self.input_shape = cfg.image_size
        # self.PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])
        self.PIXEL_MEANS = [122.7717, 115.9465, 102.9801]
        self.is_train = is_train
        self.num_classes = cfg.num_classes
        self.data_dir = cfg.data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        file_pattern = self.data_dir + "/*" + '.tfrecords'
        self.tfrecord_files = tf.gfile.Glob(file_pattern)
        self.classes = []

        if len(self.tfrecord_files) == 0:
            self.make_tfrecord(self.data_dir, cfg.tfrecord_num)
            self.tfrecord_files = tf.gfile.Glob(file_pattern)

    def read_annotations(self):
        images = []
        bboxes = []

        files = os.listdir(cfg.image_dir)
        for file_idx, file in enumerate(files):
            if file_idx % 1000 == 0:
                print(file_idx)

            if file.find('xml') != -1:
                continue

            xml_file = file.replace('png', 'xml').replace('jpg', 'xml')
            try:
                tree = ET.parse(os.path.join(cfg.image_dir, xml_file))
            except:
                print(xml_file)
                continue

            objs = tree.findall('object')

            # object가 없는 경우
            if len(objs) == 0 or objs[0].find('name') is None:
                continue

            size = tree.findall('size')[0]
            height = float(size.find('height').text)
            width = float(size.find('width').text)
            boxes = []
            for obj in objs:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                label = obj.find('name').text
                if xmin < 0 or ymin < 0:
                    print(file)
                boxes.append([xmin, ymin, xmax, ymax, label, height, width])
                self.classes.append(label)

            bboxes.append(boxes)
            images.append(os.path.join(cfg.image_dir,  file))

        self.classes = list(set(self.classes))
        self.classes.sort()
        if not os.path.exists(cfg.class_file):
            open(cfg.class_file, 'w', encoding='utf-8').writelines("\n".join(self.classes))
        return images, bboxes

    def make_tfrecord(self, tfrecord_path, num_tfrecords):
        images, bboxes = self.read_annotations()
        images_num = int(len(images) / num_tfrecords)
        for index_records in range(num_tfrecords):
            output_file = os.path.join(tfrecord_path, str(index_records) + '_' + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.gfile.FastGFile(images[index], 'rb') as file:
                        image = file.read()
                        xmin, xmax, ymin, ymax, label, height, width = [], [], [], [], [], [], []
                        for box in bboxes[index]:
                            xmin.append(box[0])
                            ymin.append(box[1])
                            xmax.append(box[2])
                            ymax.append(box[3])
                            label.append(self.classes.index(box[4]))
                            height.append(box[5])
                            width.append(box[6])

                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                                'image/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
                                'image/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
                                'image/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
                                'image/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
                                'image/label': tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                                'image/height': tf.train.Feature(float_list=tf.train.FloatList(value=height)),
                                'image/width': tf.train.Feature(float_list=tf.train.FloatList(value=width)),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 1000 == 0:
                            print('Processed {} of {} images'.format(index + 1, len(images)))
    
    def parser(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.FixedLenFeature([], dtype=tf.string),
                'image/xmin': tf.VarLenFeature(dtype=tf.float32),
                'image/xmax': tf.VarLenFeature(dtype=tf.float32),
                'image/ymin': tf.VarLenFeature(dtype=tf.float32),
                'image/ymax': tf.VarLenFeature(dtype=tf.float32),
                'image/label': tf.VarLenFeature(dtype=tf.float32),
                'image/height': tf.VarLenFeature(dtype=tf.float32),
                'image/width': tf.VarLenFeature(dtype=tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        xmin = tf.expand_dims(features['image/xmin'].values, axis=0)
        ymin = tf.expand_dims(features['image/ymin'].values, axis=0)
        xmax = tf.expand_dims(features['image/xmax'].values, axis=0)
        ymax = tf.expand_dims(features['image/ymax'].values, axis=0)
        label =features['image/label'].values
        bbox = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
        bbox = tf.transpose(bbox, [1, 0])

        height = features['image/height'].values[0]
        height = tf.cast(height, tf.float32)
        width = features['image/width'].values[0]
        width = tf.cast(width, tf.float32)

        image, bbox, scale = self.preprocess(image, bbox, height, width)

        anchors = tf.py_func(all_anchor_conner, [tf.math.ceil(width * scale), tf.math.ceil(height * scale), cfg.anchor_scales, cfg.anchor_ratios, cfg.feat_stride],
                                      tf.float32)

        # labels은 모든 anchor에 대해서 positive 1 negative는 0 나머지는 -1
        # labels_bbox 모든 anchor에 어떠한 gt object와 가장 크게 겹치는지 (anchor마다 몇 번 째 object와 겹치는지에 대한 정보)
        labels, labels_bbox = tf.py_func(anchor_labels_process,
                                         [bbox, anchors, cfg.anchor_batch, cfg.overlaps_max,
                                                           cfg.overlaps_min, tf.math.ceil(width * scale), tf.math.ceil(height * scale)]
                                         ,[tf.float32, tf.int32])

        positive_labels, positive_negative_labels, gt_positive_labels_bbox, gt_positive_negative_labels = self.generate_anchors_labels(labels, labels_bbox, bbox, anchors)
        return image, bbox, label, tf.math.ceil(height * scale), tf.math.ceil(width * scale), positive_labels, positive_negative_labels, gt_positive_labels_bbox, gt_positive_negative_labels, anchors

    def preprocess(self, image, bbox, height, width):
        min = tf.minimum(height, width)
        max = tf.maximum(height, width)
        scale_min = cfg.min_image_size / min
        scale_max = cfg.max_image_size / max

        # 먄악 image를 scale로 키웠을 대 다른 쪽이 max보다 큰 경우 있으므로
        # scale = cfg.min_image_size / min
        scale = tf.cond(scale_min * max > cfg.max_image_size, lambda : scale_max, lambda : scale_min)

        image = tf.image.resize_images(image, [tf.math.ceil(height * scale), tf.math.ceil(width * scale)],
                                               method=tf.image.ResizeMethod.BILINEAR)

        image = tf.subtract(image, self.PIXEL_MEANS)
        image = tf.divide(image, 128)
        # image -= self.PIXEL_MEANS
        # image /= 128

        xmin, ymin, xmax, ymax = tf.split(value=bbox, num_or_size_splits=4, axis=1)
        xmin = xmin * scale
        xmax = xmax * scale
        ymin = ymin * scale
        ymax = ymax * scale
        bbox = tf.concat([xmin, ymin, xmax, ymax], 1)

        if self.is_train:
            def _flip_left_right_boxes(boxes):
                # xmin, ymin, xmax, ymax, label = tf.split(value=boxes, num_or_size_splits=5, axis=1)
                xmin, ymin, xmax, ymax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
                flipped_xmin = tf.subtract(width * scale, xmax)
                flipped_xmax = tf.subtract(width * scale, xmin)
                flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax], 1)
                return flipped_boxes

            flip_left_right = tf.greater(tf.random_uniform([], dtype=tf.float32, minval=0, maxval=1), 0.5)
            image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
            bbox = tf.cond(flip_left_right, lambda: _flip_left_right_boxes(bbox), lambda: bbox)

        # image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
        # bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max=width - 1)
        # bbox = tf.cond(tf.greater(tf.shape(bbox)[0], 50), lambda: bbox[:self.50],
        #                lambda: tf.pad(bbox, paddings=[[0, self.50 - tf.shape(bbox)[0]], [0, 0]],
        #                               mode='CONSTANT'))
        return image, bbox, scale

    def generate_anchors_labels(self, labels, labels_bbox, bbox, anchors):

        # postivce anchor와 negative anchor idx 정보
        positive_negative_labels = tf.reshape(tf.where(tf.not_equal(labels, -1)), [-1])
        positive_labels = tf.reshape(tf.where(tf.equal(labels, 1)), [-1])

        # labels_gt_order
        positive_labels_bbox = tf.gather(labels_bbox, positive_labels)
        positive_labels_acnhors = tf.gather(anchors, positive_labels)
        # 실제 ground truth 좌표를 x,y,w,h로 치환
        gt_x1 = bbox[:, 0]
        gt_y1 = bbox[:, 1]
        gt_x2 = bbox[:, 2]
        gt_y2 = bbox[:, 3]

        gt_x = tf.expand_dims(tf.cast((gt_x1 + gt_x2) / 2.0, dtype=tf.float32), -1)
        gt_y = tf.expand_dims(tf.cast((gt_y1 + gt_y2) / 2.0, dtype=tf.float32), -1)
        gt_w = tf.expand_dims(tf.cast((gt_x2 - gt_x1 + 1.0), dtype=tf.float32), -1)
        gt_h = tf.expand_dims(tf.cast((gt_y2 - gt_y1 + 1.0), dtype=tf.float32), -1)
        gt_bbox = tf.concat([gt_x, gt_y, gt_w, gt_h], axis=1)

        positive_labels_acnhors_x1 = positive_labels_acnhors[:, 0]
        positive_labels_acnhors_y1 = positive_labels_acnhors[:, 1]
        positive_labels_acnhors_x2 = positive_labels_acnhors[:, 2]
        positive_labels_acnhors_y2 = positive_labels_acnhors[:, 3]

        positive_labels_acnhors_x = tf.cast((positive_labels_acnhors_x2 + positive_labels_acnhors_x1) / 2.0, dtype=tf.float32)
        positive_labels_acnhors_y = tf.cast((positive_labels_acnhors_y2 + positive_labels_acnhors_y1) / 2.0, dtype=tf.float32)
        positive_labels_acnhors_w = tf.cast((positive_labels_acnhors_x2 - positive_labels_acnhors_x1), dtype=tf.float32)
        positive_labels_acnhors_h = tf.cast((positive_labels_acnhors_y2 - positive_labels_acnhors_y1), dtype=tf.float32)

        # positive anchor에 대한 ground truth x,y,w,h 좌표
        positive_labels_bbox = tf.gather(gt_bbox, positive_labels_bbox)

        # RPN에서 bbox loss를 구할 때 사용
        # 원레 식은 postive 1, negative0 으로 하고 bbox 차이를 곱한 뒤 더하지만 posivive만 구해서 더하는 거랑 같음
        # positive achor bbox 좌표를 x,y,w,h로 치환
        # positive_labels_x1 = positive_labels_bbox[:, 0]
        # positive_labels_y1 = positive_labels_bbox[:, 1]
        # positive_labels_x2 = positive_labels_bbox[:, 2]
        # positive_labels_y2 = positive_labels_bbox[:, 3]
        # positive_labels_x = tf.cast((positive_labels_x2 + positive_labels_x1) / 2.0, dtype=tf.float32)
        # positive_labels_y = tf.cast((positive_labels_y2 + positive_labels_y1) / 2.0, dtype=tf.float32)
        # positive_labels_w = tf.cast((positive_labels_x2 - positive_labels_x1), dtype=tf.float32)
        # positive_labels_h = tf.cast((positive_labels_y2 - positive_labels_y1), dtype=tf.float32)

        # 실제 모델이 예측해야 하는 x,y,w,h 값
        gt_positive_labels_x = tf.cast((positive_labels_bbox[:, 0] - positive_labels_acnhors_x) / positive_labels_acnhors_w, dtype=tf.float32)
        gt_positive_labels_y = tf.cast((positive_labels_bbox[:, 1] - positive_labels_acnhors_y) / positive_labels_acnhors_h, dtype=tf.float32)
        gt_positive_labels_w = tf.cast(tf.log(positive_labels_bbox[:, 2] / positive_labels_acnhors_w), dtype=tf.float32)
        gt_positive_labels_h = tf.cast(tf.log(positive_labels_bbox[:, 3] / positive_labels_acnhors_h), dtype=tf.float32)

        gt_positive_labels_bbox = tf.stack(
            [gt_positive_labels_x, gt_positive_labels_y, gt_positive_labels_w, gt_positive_labels_h], axis=1)

        gt_positive_negative_labels = tf.gather(labels, positive_negative_labels)
        gt_positive_negative_labels = tf.to_int32(gt_positive_negative_labels)

        return positive_labels, positive_negative_labels, gt_positive_labels_bbox, gt_positive_negative_labels

    def build_dataset(self, batch_size):
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecord_files)
        dataset = dataset.map(self.parser, num_parallel_calls=10)

        # dataset = dataset.repeat().shuffle(1000).batch(batch_size).prefetch(batch_size)
        dataset = dataset.repeat().shuffle(1).batch(batch_size).prefetch(batch_size)

        return dataset

if __name__ == '__main__':
    dataset = preprocess(True)
    out_dir = 'F:\\aaa_'
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    # dataset.read_annotations()
    train_data = dataset.build_dataset(cfg.batch_size)
    iterator = train_data.make_one_shot_iterator()

    tf_image, tf_bbox, tf_label, tf_height, tf_width , \
    tf_positive_labels, tf_positive_negative_labels, tf_gt_positive_labels_bbox, tf_gt_positive_negative_labels, tf_anchors = iterator.get_next()

    sess = tf.Session()
    while True:
        image, bbox, label, height, width, positive_labels, positive_negative_labels, gt_positive_labels_bbox, gt_positive_negative_labels, anchors = \
            sess.run([tf_image, tf_bbox, tf_label, tf_height, tf_width , tf_positive_labels,
                      tf_positive_negative_labels, tf_gt_positive_labels_bbox, tf_gt_positive_negative_labels, tf_anchors])



        image = image.astype(np.uint8)
        image = image[0]
        bbox = bbox[0]
        label = label[0]
        anchors = all_anchor_conner(width, height, cfg.anchor_scales, cfg.anchor_ratios)

        for box_idx, box in enumerate(bbox):
            # print('box : ', box.astype(np.uint8))
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))

        print('image : ', image.shape)
        print('label : ', label.shape)
        print('anchors : ', anchors.shape)
        print('len(boxes) : ', len(bbox))
        print('positive_negative_label : ', positive_negative_labels.shape)
        print('positive_label : ', positive_labels.shape)
        print('gt_positive_label : ', gt_positive_labels_bbox.shape)
        print('gt_positive_negative_labels : ', gt_positive_negative_labels.shape)
        print('positive_labels : ', positive_labels)
        print('positive_negative_labels : ', positive_negative_labels)
        print('gt_positive_negative_labels : ', gt_positive_negative_labels)
        print('gt_positive_labels_bbox : ', gt_positive_labels_bbox)

        positive_labels_acnhors = anchors[positive_labels,]
        positive_labels_acnhors = positive_labels_acnhors[0]
        for positive_labels_acnhors_idx, positive_anchor in enumerate(positive_labels_acnhors):
            # print((int(positive_labels_acnhors[0]), int(positive_labels_acnhors[1])), (int(positive_labels_acnhors[2]), int(positive_labels_acnhors[3])))
            # print(gt_obj[gener_acs_gt_idx])
            cv2.rectangle(image, (int(positive_anchor[0]), int(positive_anchor[1])), (int(positive_anchor[2]), int(positive_anchor[3])), (0, 0, 255))

        cv2.imshow('i', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # labels, anchor_obj = anchor_labels_process(boxes, acs, cfg.anchor_batch, cfg.overlaps_max,
        #                                                    cfg.overlaps_min, w, h)
        #
        # print('width, height : ', w, h)
        # print('labels : ', labels.shape)
        # print('anchor_obj : ', anchor_obj.shape)
        #
        #
        # # anchor_obj 는 그림에서 표시할 수 있는 모든 anchor 임
        # # print(anchor_obj[labels==1].shape)
        # # 실제 label은 1은 object가 있다, 0은 배경
        # # cls loss는 2개를 다 맞혀야 하고
        # # bbox loss는 positive anchor에 대해서만 계산
        #
        # # gener_acs_gts = acs[labels!=-1, ]
        # # gt_obj = anchor_obj[labels!=-1,]
        # gener_acs_gts = acs[labels==1, ]
        #
        # # bbox는 무조건 positive anchor 에대해서만!!!
        # gt_obj = anchor_obj[labels==1,]
        #
        # print('gener_acs_gts : ', gener_acs_gts.shape)
        # print('gt_obj : ', gt_obj.shape)
        #
        # for gener_acs_gt_idx, gener_acs_gt in enumerate(gener_acs_gts):
        #     # print(gener_acs_gt)
        #     # print(gt_obj[gener_acs_gt_idx])
        #     cv2.rectangle(i, (int(gener_acs_gt[0]), int(gener_acs_gt[1])), (int(gener_acs_gt[2]), int(gener_acs_gt[3])), (0, 0, 255))
        #
        #
        # gt_x1 = boxes[:, 0]
        # gt_y1 = boxes[:, 1]
        # gt_x2 = boxes[:, 2]
        # gt_y2 = boxes[:, 3]
        #
        # re_gt_0 = (gt_x1 + gt_x2) / 2.0
        # re_gt_1 = (gt_y1 + gt_y2) / 2.0
        # re_gt_2 = (gt_x2 - gt_x1) + 1.0
        # re_gt_3 = (gt_y2 - gt_y1) + 1.0
        #
        #
        # re_gt_0 = np.reshape(re_gt_0, [-1, 1])
        # re_gt_1 = np.reshape(re_gt_1, [-1, 1])
        # re_gt_2 = np.reshape(re_gt_2, [-1, 1])
        # re_gt_3 = np.reshape(re_gt_3, [-1, 1])
        #
        # re_gt = np.concatenate([re_gt_0, re_gt_1, re_gt_2, re_gt_3], axis=1)
        #
        # print('re_gt.shape : ', re_gt.shape)
        # print('re_gt : ', re_gt)
        # print('gt_obj : ', gt_obj)
        # # positive anchor에 대한 gt 좌표
        # gt = re_gt[gt_obj]
        # print('gt : ', gt)
        # print('gt : ', gt.shape)
        #
        # anchor_x1 = gener_acs_gts[:, 0]
        # anchor_y1 = gener_acs_gts[:, 1]
        # anchor_x2 = gener_acs_gts[:, 2]
        # anchor_y2 = gener_acs_gts[:, 3]
        # re_anchor_0 = (anchor_x2 + anchor_x1) / 2.0
        # re_anchor_1 = (anchor_y2 + anchor_y1) / 2.0
        # re_anchor_2 = (anchor_x2 - anchor_x1)
        # re_anchor_3 = (anchor_y2 - anchor_y1)
        #
        # bbox_gt_0 = (gt[:, 0] - re_anchor_0) / re_anchor_2
        # bbox_gt_1 = (gt[:, 1] - re_anchor_1) / re_anchor_3
        # bbox_gt_2 = np.log(gt[:, 2] / re_anchor_2)
        # bbox_gt_3 = np.log(gt[:, 3] / re_anchor_3)
        # re_bbox_gt = np.stack(
        #     [bbox_gt_0, bbox_gt_1, bbox_gt_2, bbox_gt_3], axis=1)
        #
        # print(re_bbox_gt)
        #
        #
        # cv2.imshow('i', i)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # # cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(count)), i)
        # # count+=1
        # # if count == 100:
        # #     break