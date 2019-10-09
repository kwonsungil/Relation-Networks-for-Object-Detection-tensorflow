import os
import time
from easydict import EasyDict as edict

cfg = edict()

cfg.ROOT = 'F:\\kwon\\image-classification-master'

# cfg.out_dir = os.path.join(ROOT, 'logs', 'ResNet_101')
# now = time.time()
# cfg.summary_dir = os.path.join(cfg.out_dir, str(now), "summaries")
# cfg.checkpoint_dir = os.path.join(cfg.out_dir, str(now), "checkpoints")

cfg.imagenet_root_dir = 'F:\\kwon\\imagenet'
cfg.image_dir = os.path.join(cfg.imagenet_root_dir, 'images', 'ILSVRC2012_img_train')
cfg.label_file = os.path.join(cfg.imagenet_root_dir, 'ImageNet Label.txt')
cfg.class_file = os.path.join(cfg.ROOT, 'data', 'classes', 'imagenet_classes.txt')

# cfg.data_dir = os.path.join(ROOT, 'data', 'dataset', 'imagenet')
cfg.data_dir = os.path.join(cfg.ROOT, 'data', 'dataset', 'imagenet')

cfg.tfrecord_num = 200


cfg.image_size = 224
cfg.summary_step = 1000
cfg.num_classes = 1000
cfg.dropout_keep_prob = 0.5
cfg.initial_learning_rate = 0.01
cfg.decay_rate = 0.1
cfg.staircase = True
cfg.epsilon = 1e-3
cfg.decay = 0.99
cfg.weight_decay = 0.00004

cfg.batch_size = 48
cfg.train_num = 1281167
cfg.epochs = 100
cfg.decay_steps = 30 * (cfg.train_num // cfg.batch_size)