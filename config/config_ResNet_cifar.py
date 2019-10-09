import os
import time
from easydict import EasyDict as edict

cfg = edict()

ROOT = 'F:\\kwon\\image-classification-master'

cfg.out_dir = os.path.join(ROOT, 'logs', 'ResNet')
now = time.time()
cfg.summary_dir = os.path.join(cfg.out_dir, str(now), "summaries")
cfg.checkpoint_dir = os.path.join(cfg.out_dir, str(now), "checkpoints")

cfg.image_size = 32
cfg.summary_step = 1000
cfg.num_classes = 10
cfg.dropout_keep_prob = 0.5
cfg.initial_learning_rate = 0.001
cfg.decay_steps = 16000
cfg.decay_rate = 0.1
cfg.staircase = True
cfg.epsilon = 1e-3
cfg.decay = 0.99
cfg.weight_decay = 0.00004
