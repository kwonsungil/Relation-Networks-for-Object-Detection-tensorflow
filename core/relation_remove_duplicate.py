import tensorflow as tf
import numpy as np
from core.relation_module import RelationModule
import config


class remove_duplicate:
    def __init__(self, sample_roi, roi_score, roi_cls_loc):
        sess = tf.Session()
        self.num_class = 20 + 1
        self.loc_normalize_mean = [0., 0., 0., 0.] * self.num_class
        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2] * self.num_class

        # pooling 된 RoI feature 값들
        # [-1, 7, 7, 1024]
        # N = sample_roi.shape[0]

        # [-1, 2]
        # RoI 갯수만큼 softmax값이 들어 있음

        # roi_score
        # 1번 idx가 object가 있는 경우
        roi_range = tf.range(start=0, limit=tf.shape(roi_score)[0], delta=1)
        roi_score_arg_max = tf.argmax(roi_score, 1)
        roi_score_max = tf.reduce_max(roi_score, 1)

        # [-1, 4]
        # model이 예측한 좌표에 anchor x,y,w,h를 적용하고 x1,y1,x2,y2로 치환한 값
        # roi_cls_loc

        # object가 있다고 예측한 RoI
        # 0 은 없다고 예측한 ROI
        not_zero = tf.where(tf.not_equal(roi_score_arg_max, 0))
        print(sess.run([roi_score_arg_max, roi_range, not_zero]))
        not_zero = tf.squeeze(not_zero, 1)
        # not_zero = tf.not_equal(roi_score_arg_max, 0)

        #TODO
        # RoI가 모두 0일 경우 제거하기 위해서
        # not_zero = tf.cond(tf.equal(tf.shape(not_zero)[0], 0), lambda: None, lambda: not_zero)

        # Object가 있다고 판단한 경우의 feature들만 가져옴
        roi_cls_loc = tf.gather(roi_cls_loc, not_zero)
        roi_score_max = tf.gather(roi_score_max, not_zero)
        # not_zero는 object가 있다고 예측한 idx 정보
        # roi_score_arg_max = tf.gather(roi_score_arg_max, not_zero)
        sample_roi = tf.gather(sample_roi, not_zero)

        sorted_idx = tf.argsort(roi_score_max, direction='DESCENDING')

        # Object가 있든 feature들을 score값으로 sorting
        roi_cls_loc = tf.gather(roi_cls_loc, sorted_idx, axis=0)
        roi_score_max = tf.gather(roi_score_max, sorted_idx, axis=0)
        not_zero = tf.gather(not_zero, sorted_idx, axis=0)
        sample_roi = tf.gather(sample_roi, sorted_idx, axis=0)

        roi_score_v, not_zero_v, sample_roi_v, sorted_idx_v, roi_score_max_v = sess.run([roi_score,
                                                                                             not_zero,
                                                                                             sample_roi,
                                                                                             sorted_idx,
                                                                                             roi_score_max,
                                                                                             ])
        print('roi_score_v : ', roi_score_v[:10])
        print('not_zero_v : ', not_zero_v[:10])
        print('not_zero_v : ', not_zero_v.shape)
        print('sample_roi_v', sample_roi_v.shape)
        print('sorted_idx_v', sorted_idx_v[:10])
        # sorting 된 socre
        print('roi_score_max_v', roi_score_max_v[:10])
        print('roi_score_max_v', roi_score_max_v.shape)

        # [numver of RoI, 1024] (score 값이 zero가 없을 때)
        # rank dim은 후보 feature의 개수 많큼 생성해야함
        nms_rank_embedding = self.RankEmbedding(rank_dim=tf.shape(sample_roi)[0])
        # backpropagation을 해야 할까...? 논문에서는 아닌 거 같음
        nms_rank = tf.layers.dense(nms_rank_embedding, 128, use_bias=True,
                                   kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                   activation=tf.nn.relu)

        roi_feat_embedding = tf.layers.dense(sample_roi, 128, use_bias=True,
                                             kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                             activation=tf.nn.relu)

        nms_embedding_feat = tf.add(nms_rank, roi_feat_embedding)
        # sorting된 bbox정보를 넣고 positional embedding을 구함
        nms_logit = RelationModule(nms_embedding_feat, roi_cls_loc)

        print('nms_rank_embedding : ', nms_rank_embedding)
        print('nms_rank : ', nms_rank)
        print('roi_feat_embedding : ', roi_feat_embedding)
        print('nms_embedding_feat : ', nms_embedding_feat)
        print('nms_logit.result : ', nms_logit.result)

        s1 = tf.layers.dense(nms_logit.result, 1, use_bias=True,
                                    kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
                                    activation=tf.nn.sigmoid)

        nms_scores = s1 * roi_score_max
        # print('s1 : ', s1)
        # print('roi_score_max : ', roi_score_max)
        # print('nms_scores : ', nms_scores)

        self.nms_scores = nms_scores
        self.not_zero = not_zero
        self.roi_cls_loc = roi_cls_loc

        sess.run(tf.global_variables_initializer())
        a, b, c = sess.run([nms_scores, not_zero, roi_cls_loc])
        print(a.shape)
        print(b.shape)
        print(c.shape)



    def RankEmbedding(self, rank_dim, feat_dim=1024, wave_len=1000.):
        # # numpy implementation
        # rank_range = np.arange(0, rank_dim)
        # feat_range = np.arange(feat_dim / 2)
        # dim_mat = feat_range / (feat_dim / 2)
        # dim_mat = 1. / (np.power(wave_len, dim_mat))
        #
        # # (1, 512)
        # dim_mat = np.reshape(dim_mat, [1, -1])
        # # (128, 1)
        # rank_mat = np.reshape(rank_range, [-1, 1])
        # # (128, 512)
        # mul_mat = rank_mat * dim_mat
        # sin_mat = np.sin(mul_mat)
        # cos_mat = np.cos(mul_mat)
        # # [128, 1024]
        # embedding = np.concatenate([sin_mat, cos_mat], -1)

        rank_range = tf.range(start=0, limit=rank_dim, delta=1)
        feat_range = tf.range(start=0, limit=feat_dim / 2, delta=1)
        dim_mat = feat_range / (feat_dim / 2)
        dim_mat = 1. / (tf.pow(wave_len, dim_mat))

        # (1, 512)
        dim_mat = tf.reshape(dim_mat, [1, -1])
        # (128, 1)
        rank_mat = tf.reshape(rank_range, [-1, 1])
        # (128, 512)
        mul_mat = tf.to_float(rank_mat) * dim_mat
        sin_mat = tf.sin(mul_mat)
        cos_mat = tf.cos(mul_mat)
        # [128, 1024]
        embedding = tf.concat([sin_mat, cos_mat], -1)

        return embedding


if __name__ == '__main__':

    sample_roi = tf.random_normal(shape=[256, 7, 7, 1024])
    roi_score = tf.random_normal(shape=[256, 2])
    roi_cls_loc = tf.random_normal(shape=[256, 4])

    sample_roi = tf.layers.flatten(sample_roi)
    rm = remove_duplicate(sample_roi, roi_score, roi_cls_loc)
