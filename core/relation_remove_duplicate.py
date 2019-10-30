import tensorflow as tf


class remove_duplicate:
    def __init__(self, appearance_feature, geometric_feature, sample_roi, roi_score, roi_cls_loc, is_duplicated=False):
        # sess = tf.Session()
        # number of relation
        self.Nr = 16
        # appearance feature dimension
        self.appearance_feature = 1024
        # geo feature dimension
        self.geo_feature = 64
        # key feature dimension
        self.key_feature = 64

        self.num_class = 20 + 1
        self.loc_normalize_mean = [0., 0., 0., 0.] * self.num_class
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2] * self.num_class

        # pooling 된 RoI feature 값들
        # [-1, 7, 7, 1024]
        # N = sample_roi.shape[0]
        N = 256

        # [-1, 1]
        # RoI 갯수만큼 softmax값이 들어 있음
        # roi_score

        # [-1, 4]
        # model이 예측한 좌표에 anchor x,y,w,h를 적용하고 x1,y1,x2,y2로 치환한 값
        # roi_cls_loc

        # roi = at.totensor(sample_roi)

        # zero가 아닌 idx
        not_zero = tf.where(tf.not_equal(roi_score, 0))
        not_zero_size = tf.shape(not_zero)[0]

        # RoI가 모두 0일 경우 제거하기 위해서
        not_zero = tf.cond(not_zero_size==0, lambda: None, lambda: not_zero)

        if not_zero is not None:
            roi_cls_loc = roi_cls_loc[not_zero]
            roi_score = roi_score[not_zero]
            sample_roi = sample_roi[not_zero]


        # score 순으로 정렬 필요
        range = tf.range(start=0, limit=N, delta=1)
        print(range)
        roi_score_argmax = tf.argmax(roi_score, axis=1)
        print(roi_score_argmax)
        # print(sess.run([roi_score_argmax]))



        # self.NMS_rank = tf.layers.dense(appearance_feature, 128, use_bias=True,
        #                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        #                            activation=tf.nn.relu)
        # self.NMS_logit = tf.layers.dense(appearance_feature, 1, use_bias=True,
        #                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        #                                 activation=tf.nn.relu)
        # self.RoI = tf.layers.dense(appearance_feature, 128, use_bias=True,
        #                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        #                            activation=tf.nn.relu)
        # result = []
        # for num in range(self.Nr):
        #     result.append(self.relation(appearance_feature, geometric_feature))
        #
        # result = tf.concat(result, axis=1)

    def relation(self, appearance_feature, geometric_feature):

        # (number of RoI * 7 * 7 * 1024,, 1024)
        num_roi = appearance_feature.shape[0]
        Wk = tf.layers.dense(appearance_feature, self.key_feature, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)
        Wq = tf.layers.dense(appearance_feature, self.key_feature, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)
        Wv = tf.layers.dense(appearance_feature, self.key_feature, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)
        Wg = tf.layers.dense(tf.nn.relu(self.PositionalEmbedding(geometric_feature)), 1, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)

        Wk = tf.reshape(Wk, [1, num_roi, self.key_feature])
        Wq = tf.reshape(Wq, [num_roi, 1, self.key_feature])
        print('Wk : ', Wk)
        print('Wq : ', Wq)
        scaled_dot = tf.reduce_sum([Wk * Wq], axis=-1) / tf.sqrt(tf.to_float(self.key_feature))
        # scaled_dot = Wk * Wq / tf.sqrt(tf.to_float(self.key_feature))
        print('scaled_dot : ', scaled_dot)
        print('Wg : ', Wg)
        Wg = tf.reshape(Wg, [num_roi, num_roi])
        print('Wg : ', Wg)
        Wa = tf.reshape(scaled_dot, [num_roi, num_roi])

        Wmn = tf.log(tf.clip_by_value(Wg, clip_value_min=1e-6, clip_value_max=100000000)) + Wa
        Wmn = tf.nn.softmax(Wmn, axis=1)
        print('Wmn : ', Wmn)
        Wmn = tf.reshape(Wmn, [num_roi, num_roi, 1])
        print('Wmn : ', Wmn)
        print('Wv : ', Wv)
        Wv = tf.reshape(Wv, [num_roi, 1, -1])
        print('Wv : ', Wv)

        return tf.reduce_sum(Wmn * Wv, axis=-2)

    def PositionalEmbedding(self, geometric_feature, dim_g=64, wave_len=1000.):
        # (number of RoI, coordinate of Roi)
        # x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=1)
        xmin, ymin, xmax, ymax = tf.split(value=geometric_feature, num_or_size_splits=4, axis=1)

        x = (xmin + xmax) * 0.5
        y = (ymin + ymax) * 0.5
        w = (xmax - xmin) + 1.
        h = (ymax - ymin) + 1.

        delta_x = x - tf.reshape(x, [1, -1])
        print(delta_x)
        # delta_x = torch.clamp(tf.abs(delta_x / w), min=1e-3)
        # delta_x = torch.log(delta_x)
        delta_x = tf.clip_by_value(tf.abs(delta_x / w), clip_value_min=1e-3, clip_value_max=100000000)
        delta_x = tf.log(delta_x)

        delta_y = y - tf.reshape(y, [1, -1])
        delta_y = tf.clip_by_value(tf.abs(delta_y / h), clip_value_min=1e-3, clip_value_max=100000000)
        delta_y = tf.log(delta_y)

        delta_w = tf.log(w / tf.reshape(w, [1, -1]))
        delta_h = tf.log(h / tf.reshape(h, [1, -1]))
        shape = delta_h.shape

        delta_x = tf.reshape(delta_x, [shape[0], shape[1], 1])
        print(delta_x)
        delta_y = tf.reshape(delta_y, [shape[0], shape[1], 1])
        delta_w = tf.reshape(delta_w, [shape[0], shape[1], 1])
        delta_h = tf.reshape(delta_h, [shape[0], shape[1], 1])

        position_mat = tf.concat([delta_x, delta_y, delta_w, delta_h], axis=-1)

        # feat_range = torch.arange(dim_g / 8).cuda()
        feat_range = tf.range(dim_g / 8)
        dim_mat = feat_range / (dim_g / 8)
        # dim_mat = tf.to_int32(dim_mat)
        print(dim_mat)
        dim_mat = 1 / (tf.pow(wave_len, dim_mat))

        dim_mat = tf.reshape(dim_mat, [1, 1, 1, -1])
        position_mat = tf.reshape(position_mat, [shape[0], shape[1], 4, -1])
        # dim_mat = dim_mat.view(1, 1, 1, -1)
        # position_mat = position_mat.view(shape[0], shape[1], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = tf.reshape(mul_mat, [shape[0], shape[1], -1])
        # mul_mat = mul_mat.view(shape[0], shape[1], -1)
        sin_mat = tf.sin(mul_mat)
        cos_mat = tf.cos(mul_mat)
        embedding = tf.concat((sin_mat, cos_mat), -1)

        return embedding

    def clip_boxes(self, boxes, img_width, img_height):
          img_width = tf.cast(img_width, tf.float32)
          img_height = tf.cast(img_height, tf.float32)
          b0 = tf.maximum(tf.minimum(boxes[:, 0], img_width - 1), 0.0)
          b1 = tf.maximum(tf.minimum(boxes[:, 1], img_height - 1), 0.0)
          b2 = tf.maximum(tf.minimum(boxes[:, 2], img_width - 1), 0.0)
          b3 = tf.maximum(tf.minimum(boxes[:, 3], img_height - 1), 0.0)
          return tf.stack([b0, b1, b2, b3], axis=1)


if __name__ == '__main__':
    appearance_feature = tf.random_normal(shape=[256, 7, 7, 1024])
    geometric_feature = tf.random_normal(shape=[256, 4])
    sample_roi = tf.random_normal(shape=[256, 7, 7, 1024])
    roi_score = tf.random_normal(shape=[256, 1])
    roi_cls_loc = tf.random_normal(shape=[256, 4])

    appearance_feature = tf.layers.flatten(appearance_feature)
    rm = remove_duplicate( appearance_feature, geometric_feature, sample_roi, roi_score, roi_cls_loc)
