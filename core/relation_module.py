import tensorflow as tf
import config as cfg

class RelationModule:
    def __init__(self, appearance_feature, geometric_feature, is_duplicated = False):
        # number of relation
        self.Nr = 16
        # appearance feature dimension
        self.appearance_feature_dim = 1024
        # geo feature dimension
        self.geo_feature_dim = 64
        # key feature dimension
        self.key_feature_dim = 64

        result = []
        for num in range(self.Nr):
            # appearance_feature는 list 형태로 들어가 있음
            result.append(self.relation(appearance_feature, geometric_feature))

        self.result = tf.concat(result, axis=1)
        print(self.result)

    def relation(self, appearance_feature, geometric_feature):
        # (number of RoI * 7 * 7 * 1024,, 1024)
        num_roi = appearance_feature.shape[0]
        # num_roi = cfg.anchor_batch
        Wk = tf.layers.dense(appearance_feature, self.key_feature_dim, use_bias = True,
                                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                activation=tf.nn.relu)
        Wq = tf.layers.dense(appearance_feature, self.key_feature_dim, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)
        Wg = tf.layers.dense(tf.nn.relu(self.PositionalEmbedding(geometric_feature)), 1, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)
        print('Wg :  ', Wg)
        Wv = tf.layers.dense(appearance_feature, self.key_feature_dim, use_bias=True,
                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                             activation=tf.nn.relu)

        Wk = tf.reshape(Wk, [1, num_roi, self.key_feature_dim])
        Wq = tf.reshape(Wq, [num_roi, 1, self.key_feature_dim])
        print('Wk : ', Wk)
        print('Wq : ', Wq)
        scaled_dot = tf.reduce_sum([Wk * Wq], axis=-1) / tf.sqrt(tf.to_float(self.key_feature_dim))
        # scaled_dot = Wk * Wq / tf.sqrt(tf.to_float(self.key_feature_dim))
        print('scaled_dot : ', scaled_dot)
        print('Wg : ', Wg)
        Wg = tf.reshape(Wg, [num_roi, num_roi])
        print('Wg : ', Wg)
        Wa = tf.reshape(scaled_dot, [num_roi, num_roi])

        Wmn = tf.log(tf.clip_by_value(Wg, clip_value_min = 1e-6, clip_value_max=100000000)) + Wa
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
        xmin, ymin, xmax, ymax = tf.split(value=geometric_feature, num_or_size_splits=4, axis=1)

        x = (xmin + xmax) * 0.5
        y = (ymin + ymax) * 0.5
        w = (xmax - xmin) + 1.
        h = (ymax - ymin) + 1.

        delta_x = x - tf.reshape(x, [1, -1])
        delta_x = tf.clip_by_value(tf.abs(delta_x / w), clip_value_min=1e-3, clip_value_max=100000000)
        delta_x = tf.log(delta_x)

        delta_y = y - tf.reshape(y, [1, -1])
        delta_y = tf.clip_by_value(tf.abs(delta_y / h), clip_value_min=1e-3, clip_value_max=100000000)
        delta_y = tf.log(delta_y)

        delta_w = tf.log(w / tf.reshape(w, [1, -1]))
        delta_h = tf.log(h / tf.reshape(h, [1, -1]))
        shape = tf.shape(delta_h)

        print('delta_w : ', delta_w)
        print('delta_h : ', delta_h)
        delta_x = tf.reshape(delta_x, [shape[0], shape[1], 1])
        delta_y = tf.reshape(delta_y, [shape[0], shape[1], 1])
        delta_w = tf.reshape(delta_w, [shape[0], shape[1], 1])
        delta_h = tf.reshape(delta_h, [shape[0], shape[1], 1])

        position_mat = tf.concat([delta_x, delta_y, delta_w, delta_h], axis=-1)

        # feat_range = torch.arange(dim_g / 8).cuda()
        feat_range = tf.range(dim_g / 8)
        dim_mat = feat_range / (dim_g / 8)
        # dim_mat = tf.to_int32(dim_mat)
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
        print('embedding : ', embedding)
        embedding.set_shape([None, None, 64])

        return embedding



if __name__ == '__main__':
    appearance_feature = tf.random_normal(shape=[256, 7, 7, 1024])
    geometric_feature = tf.random_normal(shape=[256, 4])

    appearance_feature = tf.layers.flatten(appearance_feature)
    print(appearance_feature)
    rm = RelationModule(appearance_feature, geometric_feature)