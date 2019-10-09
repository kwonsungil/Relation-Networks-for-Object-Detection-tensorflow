import tensorflow as tf
import numpy as np
import os
import pickle
import datetime
from config.config_ResNet_cifar import cfg


class ResNet:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.graph = tf.Graph()
        ConfigProto = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
        ConfigProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=ConfigProto, graph=self.graph)

        self.kernel_regularizer = tf.contrib.layers.l2_regularizer(cfg.weight_decay)
        self.kernel_initializer = tf.contrib.layers.xavier_initializer()

        with self.graph.as_default():
            self.logits = self.build_network()
            if self.is_train:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.learning_rate = tf.train.exponential_decay(
                    cfg.initial_learning_rate, self.global_step, cfg.decay_steps,
                    cfg.decay_rate, cfg.staircase, name='learning_rate')
                self.loss, self.accuracy = self.compute_loss()

                self.summary_writer = tf.summary.FileWriter(cfg.summary_dir, graph=self.sess.graph)
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        # self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,
                                                                                        global_step=self.global_step,
                                                                                        name='optimizer')
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar("acc", self.accuracy)
                tf.summary.scalar("lr", self.learning_rate)
                self.summary_op = tf.summary.merge_all()

                os.makedirs(cfg.summary_dir, exist_ok=True)
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=40)

            if len(os.listdir(cfg.out_dir)) > 0:
                pre_ckpt_dir = os.listdir(cfg.out_dir).pop()
                filename = tf.train.latest_checkpoint(os.path.join(cfg.out_dir, pre_ckpt_dir, 'checkpoints'))
                if filename is not None:
                    print('restore from : ', filename)
                    self.saver.restore(self.sess, filename)

    def build_network(self):
        self.input_x = tf.placeholder(tf.float32, [None, cfg.image_size, cfg.image_size, 3], name="input_x")
        self.label = tf.placeholder(tf.int32, [None, cfg.num_classes], name="label")
        if self.is_train:
            # self.input_x = tf.image.random_flip_left_right(self.input_x)
            self.input_x = tf.image.resize_image_with_crop_or_pad(
                tf.pad(self.input_x, np.array([[0, 0], [2, 2], [2, 2], [0, 0]]), name='random_crop_pad'),
                cfg.image_size, cfg.image_size)

        with tf.variable_scope('ResNet'):
            # https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0
            model = tf.layers.conv2d(inputs=self.input_x, filters=16, kernel_size=(3, 3), strides=(1, 1),
                                     padding='SAME', name='conv_1', kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
            model = tf.layers.batch_normalization(inputs= model)
            model = tf.nn.relu(model)

            # 첫번째 residual block 만 stride 1
            model = self.residual_bottleneck('layers_2n', model, filters=16, kernel_size=3, stride=1)
            print('layers_1 : ', model)
            model = self.residual_bottleneck('layers_4n', model, filters=32, kernel_size=3, stride=2)
            print('layers_2 : ', model)
            model = self.residual_bottleneck('layers_6n', model, filters=64, kernel_size=3, stride=2)
            print('layers_3 : ', model)
            model = self.global_avg_pool('gobal_avg_pool', model)
            print('gobal_avg_pool : ', model)
            model = tf.reshape(model, [-1, int(model.shape[1]) * int(model.shape[2]) * int(model.shape[3])])
            # model = self.fc_layer('fc', model, cfg.num_classes, activate=None)
            model = tf.layers.dense(model, cfg.num_classes)
            print('fc_layer : ', model)

            return model

    def global_avg_pool(self, name, inputs, stride=1):
        return tf.layers.average_pooling2d(inputs, pool_size=(int(inputs.shape[1]), int(inputs.shape[2])), strides=(stride, stride), padding='VALID', name=name)


    def residual_bottleneck(self, name, inputs, kernel_size, filters, stride):
        # orginal resnet은 stride가 첫 번째 convolution
        # 추후에는 정보량 손실로 인해 두 번재 convolution에 stride
        with tf.variable_scope(name):
            # 1x1
            block = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1, 1), strides=(1, 1),
                                     padding='SAME',
                                     name='conv_1', kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer)
            block = tf.layers.batch_normalization(inputs=block)
            block = tf.nn.relu(block)

            # 3x3
            block = tf.layers.conv2d(inputs=block, filters=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                                     padding='SAME',
                                     name='conv_2', kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer)
            block = tf.layers.batch_normalization(inputs=block)
            block = tf.nn.relu(block)

            # 1*1
            block = tf.layers.conv2d(inputs=block, filters=filters*4, kernel_size=(1, 1), strides=(1, 1),
                                     padding='SAME',
                                     name='conv_3', kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer)
            block = tf.layers.batch_normalization(inputs=block)
            block = tf.nn.relu(block)

            # shortcut
            if int(block.shape[3]) != int(inputs.shape[3]):
                # inputs = tf.pad(inputs, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_1')
                inputs = tf.layers.conv2d(inputs, filters=filters * 4, kernel_size=(1, 1),
                                          strides=stride, padding='VALID')

            return tf.nn.relu(block + inputs)

    def compute_loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label))
        # tf.add_to_collection('losses', loss)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.label, axis=1)), tf.float32),
            name='accuracy')

        # return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy
        return loss, accuracy

    def train(self, batch_x, batch_y, save=False):
        # print(self.graph)
        with self.graph.as_default():
            feed_dict = {self.input_x: batch_x,
                         self.label: batch_y}

            _, global_step, summary_str, loss, acc = self.sess.run(
                [self.train_op, self.global_step, self.summary_op, self.loss, self.accuracy],
                feed_dict=feed_dict)
            if save:
                self.summary_writer.add_summary(summary_str, global_step=global_step)
                self.saver.save(self.sess, os.path.join(cfg.checkpoint_dir, 'model.ckpt'), global_step=global_step)

            return global_step, loss, acc

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

