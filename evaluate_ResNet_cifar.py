import numpy as np
import os
import pickle
from models.ResNet_cifar import ResNet
from utils.load_cifar import load_test


if __name__ == '__main__':
    num_classes = 10
    batch_size = 128
    test_x, test_y = load_test(num_classes)

    net = ResNet(False)
    test_set_len = test_x.shape[0]
    total_batch = int(test_set_len / batch_size)
    total_equal = 0
    for i in range(total_batch + 1):
        if ((i + 1) * batch_size) > test_set_len:
            break

        batch_x = test_x[i * batch_size: (i + 1) * batch_size]
        batch_y = test_y[i * batch_size: (i + 1) * batch_size]

        equal = net.predcit(batch_x, batch_y)
        total_equal += equal[0]

    print('test accuracy : %.3f' % (total_equal / test_set_len))

