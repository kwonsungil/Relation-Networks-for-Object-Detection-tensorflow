import os
import tensorflow as tf
from config.config_ResNet_imagenet import cfg
import random


class preprocess:
    def __init__(self):
        self.cfg = cfg
        # self.input_shape = self.cfg.image_size
        self.num_classes = self.cfg.num_classes
        self.PIXEL_MEANS = [122.7717, 115.9465, 102.9801]
        self.data_dir = self.cfg.data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        file_pattern = self.data_dir + "/*" + '.tfrecords'
        self.tfrecord_files = tf.gfile.Glob(file_pattern)
        # self.class_map = self.get_class(self.cfg.label_file)

        if not os.path.exists(cfg.class_file):
            self.classes = os.listdir(cfg.image_dir)
            self.classes.sort()
            open(cfg.class_file, 'w', encoding='utf-8').writelines("\n".join(self.classes))
        else:
            self.classes = open(cfg.class_file, 'r', encoding='utf-8').read().splitlines()

        if len(self.tfrecord_files) == 0:
            self.make_tfrecord(self.data_dir, self.cfg.tfrecord_num)
            self.tfrecord_files = tf.gfile.Glob(file_pattern)

    def get_class(self, class_file):
        lines = open(class_file, 'r', encoding='utf-8').read().splitlines()
        class_map = {}
        for line in lines:
            line = line.strip()
            infos = line.split(' ')
            key = infos[0]
            infos.remove(key)
            value = ' '.join(infos)
            class_map[key] = value
        return class_map

    def read_annotations(self):
        labels = []
        images = []

        labels_dir = os.listdir(self.cfg.image_dir)
        for label_dir in labels_dir:
            print(self.classes.index(label_dir))
            files = os.listdir(os.path.join(self.cfg.image_dir, label_dir))
            for file_idx, file in enumerate(files):

                labels.append(self.classes.index(label_dir))
                images.append(os.path.join(self.cfg.image_dir, label_dir,  file))

        temp = list(zip(images, labels))
        random.shuffle(temp)
        images, labels = zip(*temp)

        return images, labels

    def make_tfrecord(self, tfrecord_path, num_tfrecords):
        images, labels = self.read_annotations()
        images_num = int(len(images) / num_tfrecords)
        for index_records in range(num_tfrecords):
            output_file = os.path.join(tfrecord_path, str(index_records) + '_' + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.gfile.FastGFile(images[index], 'rb') as file:
                        image = file.read()
                        label = labels[index]
                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                                'image/label': tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
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
                'image/label': tf.VarLenFeature(dtype=tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # label = tf.expand_dims(features['image/label'].values, axis=0)
        label = features['image/label'].values[0]
        label = tf.cast(label, tf.int32)
        image, label = self.preprocess(image, label)
        print(image)
        print(label)
        return image, label

    def preprocess(self, image, label):
        # image = tf.subtract(image, self.PIXEL_MEANS)
        # image = tf.divide(image, 128.)
        image -= self.PIXEL_MEANS
        image /= 128
        # imagenet에서는 4pixel paddiang and random crop 적용
        image = tf.pad(tf.expand_dims(image, axis=0), [[0, 0], [2, 2], [2, 2], [0, 0]])
        image = tf.squeeze(image, axis=0)
        image = tf.image.resize_image_with_crop_or_pad(image, cfg.image_size, cfg.image_size)
        # image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
        # image = tf.cast(image,  dtype=tf.float32)
        label = tf.one_hot(label, depth=cfg.num_classes)
        return image, label

    def build_dataset(self, batch_size):
        dataset = tf.data.TFRecordDataset(filenames=self.tfrecord_files)
        dataset = dataset.map(self.parser, num_parallel_calls=10)

        dataset = dataset.repeat().shuffle(1000).batch(batch_size).prefetch(batch_size)
        # dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)

        return dataset