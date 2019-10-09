import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
from config.config_ResNet_imagenet import cfg
from utils.load_imagenet import preprocess
import time


class ResNet:
    def __init__(self, model='resnet101', is_train=True):
        out_dir = os.path.join(cfg.ROOT, 'logs', model)
        now = time.time()
        self.summary_dir = os.path.join(out_dir, str(now), "summaries")
        self.checkpoint_dir = os.path.join(out_dir, str(now), "checkpoints")

        file_pattern = cfg.data_dir + "/*" + '.tfrecords'
        self.tfrecord_files = tf.gfile.Glob(file_pattern)
        self.is_train = is_train
        self.graph = tf.Graph()
        ConfigProto = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1))
        ConfigProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=ConfigProto, graph=self.graph)

        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(cfg.weight_decay)
        self.kernel_initializer = tf.contrib.layers.xavier_initializer()
        # self.kernel_regularizer = None
        # self.kernel_initializer = None


        # 각각의 backbone 마다 bottleneck 개수
        if model == 'resnet50':
            self.num_blocks = [3, 4, 6, 3]
        elif model == 'resnet101':
            self.num_blocks = [3, 4, 23, 3]
        elif model == 'resnet152':
            self.num_blocks = [3, 8, 36, 3]
        else:
            raise NotImplementedError

        with self.graph.as_default():
            dataset = preprocess()
            train_data = dataset.build_dataset(cfg.batch_size)
            iterator = train_data.make_one_shot_iterator()
            self.input_x, self.label = iterator.get_next()
            self.logits = self.build_network()

            save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="ResNet")
            self.saver = tf.train.Saver(var_list=save_vars, max_to_keep=10)

            if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
                logs = os.listdir(out_dir)
                logs.sort()
                pre_ckpt_dir = logs.pop()
                filename = tf.train.latest_checkpoint(os.path.join(out_dir, pre_ckpt_dir, 'checkpoints'))
            else:
                filename = None

            if self.is_train:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.learning_rate = tf.train.exponential_decay(
                    cfg.initial_learning_rate, self.global_step, cfg.decay_steps,
                    cfg.decay_rate, cfg.staircase, name='learning_rate')
                self.loss, self.accuracy = self.compute_loss()

                self.summary_writer = tf.summary.FileWriter(self.summary_dir, graph=self.sess.graph)
                with tf.control_dependencies(save_vars):
                    self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        global_step=self.global_step,
                                                                                        name='optimizer')
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar("acc", self.accuracy)
                tf.summary.scalar("lr", self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                os.makedirs(self.summary_dir, exist_ok=True)
                os.makedirs(self.checkpoint_dir, exist_ok=True)

            self.sess.run(tf.global_variables_initializer())

            # if filename is not None:
            #     print('restore from : ', filename)
            #     self.saver.restore(self.sess, filename)
            # else:
            #     print('initialize....')

    def build_network(self):
        # self.input_x = tf.placeholder(tf.float32, [None, cfg.image_size, cfg.image_size, 3], name="input_x")
        # self.label = tf.placeholder(tf.int32, [None, cfg.num_classes], name="label")

        with tf.variable_scope('ResNet'):
            model = tf.layers.conv2d(inputs=self.input_x, filters=64, kernel_size=(7, 7), strides=(2, 2),
                                     padding='SAME', name='conv_1', kernel_initializer=tf.contrib.layers.xavier_initializer(),
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
            with tf.variable_scope('layers_8n'):
                for idx in range(self.num_blocks[3]):
                    block_name = 'layers_8n_{}'.format(idx)
                    if idx == 0:
                        model = self.residual_bottleneck(block_name, model, filters=512, kernel_size=3, stride=2)
                    else:
                        model = self.residual_bottleneck(block_name, model, filters=512, kernel_size=3, stride=1)
                    print(block_name, model)

            model = self.global_avg_pool('gobal_avg_pool', model)
            print('gobal_avg_pool : ', model)
            model = tf.reshape(model, [-1, int(model.shape[1]) * int(model.shape[2]) * int(model.shape[3])])
            # model = self.fc_layer('fc', model, cfg.num_classes, activate=None)
            model = tf.layers.dense(model, cfg.num_classes)
            print('fc_layer : ', model)

        return model

    def global_avg_pool(self, name, inputs, stride=1):
        return tf.layers.average_pooling2d(inputs, pool_size=(int(inputs.shape[1]), int(inputs.shape[2])),
                                           strides=(stride, stride), padding='VALID', name=name)

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
                inputs = tf.layers.conv2d(inputs, filters=filters * 4, kernel_size=(1, 1),
                                          strides=stride, padding='VALID')

            return tf.nn.relu(block + inputs)

    def compute_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label))
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.label, axis=1)), tf.float32),
            name='accuracy')

        # return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy
        return loss, accuracy

    # def train(self, batch_x, batch_y, save=False):
    def train(self, save=False):
        with self.graph.as_default():
            # feed_dict = {self.input_x: batch_x,
            #              self.label: batch_y}
            _, global_step, summary_str, loss, acc, lr = self.sess.run(
                [self.train_op, self.global_step, self.summary_op, self.loss, self.accuracy, self.learning_rate],
            )
            if save:
                self.summary_writer.add_summary(summary_str, global_step=global_step)
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            return global_step, loss, acc, lr

    def validate(self, batch_x, batch_y):
        with self.graph.as_default():
            feed_dict = {self.input_x: batch_x,
                         self.label: batch_y}
            # ox, accuracy, loss = self.sess.run([self.ox, self.accuracy, self.total_loss], feed_dict=feed_dict)
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            return loss, acc

    def predcit(self, batch_x, batch_y):
        with self.graph.as_default():
            equal = tf.reduce_sum(
                tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.label, axis=1)), tf.float32))

            feed_dict = {self.input_x: batch_x,
                         self.label: batch_y}
            # ox, accuracy, loss = self.sess.run([self.ox, self.accuracy, self.total_loss], feed_dict=feed_dict)
            equal = self.sess.run([equal], feed_dict=feed_dict)
            return equal