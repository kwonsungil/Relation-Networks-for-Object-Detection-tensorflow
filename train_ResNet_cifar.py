import numpy as np
import os
import pickle
from models.ResNet_cifar import ResNet
from utils.load_cifar import load_train


if __name__ == '__main__':
    nnum_classes = 10
    batch_size = 128
    epochs = 100

    train_x, train_y = load_train(nnum_classes)

    net = ResNet(True)
    train_set_len = train_x.shape[0]
    r_idx = np.arange(train_x.shape[0])
    total_batch = int(train_set_len / batch_size)
    for epoch in range(epochs):
        # print(train_x[0])

        r_idx = np.arange(train_x.shape[0])
        np.random.shuffle(r_idx)
        train_x = train_x[r_idx]
        train_y = train_y[r_idx]

        for i in range(total_batch + 1):
            if ((i + 1) * batch_size) > train_set_len:
                break

            batch_x = train_x[i * batch_size: (i + 1) * batch_size]
            batch_y = train_y[i * batch_size: (i + 1) * batch_size]

            if i % 100 == 0:
                global_step, train_loss, train_acc = net.train(batch_x, batch_y, True)
                print('%d step\ttrain loss : %.3f\ttrain accuracy : %.3f' % (global_step, train_loss, train_acc))
                # val_loss, val_acc = net.validate(test_x[:200], test_y[:200])
                # print('%d step\ttrain loss : %.3f\ttrain accuracy : %.3f\tval loss : %.3f\tval accuracy : %.3f' % (
                # global_step, train_loss, train_acc, val_loss, val_acc))
            else:
                _, loss, ac = net.train(batch_x, batch_y, False)

