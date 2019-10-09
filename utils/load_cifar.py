import numpy as np
import os
import pickle

def load_train(num_classes):
    train_x = None
    train_y = None

    folder_name = "./data/dataset/cifar_10"
    for i in range(1, 6):
        f = open(os.path.join(folder_name, 'data_batch_' + str(i)), 'rb')
        datadict = pickle.load(f, encoding='latin1')

        datas = datadict["data"]
        labels = np.array(datadict['labels'])
        labels = np.eye(num_classes)[labels]

        datas = datas / 255.0
        datas = datas.reshape([-1, 3, 32, 32])
        datas = datas.transpose([0, 2, 3, 1])

        if train_x is None:
            train_x = datas
            train_y = labels
        else:
            train_x = np.concatenate((train_x, datas), axis=0)
            train_y = np.concatenate((train_y, labels), axis=0)

        f.close()

    return train_x, train_y


def load_test(num_classes):
    folder_name = "./data/dataset/cifar_10"

    f = open(os.path.join(folder_name, 'test_batch'), 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()

    test_x = datadict["data"]
    test_y = np.array(datadict['labels'])
    test_y = np.eye(num_classes)[test_y]

    test_x = np.array(test_x) / 255.0
    test_x = test_x.reshape([-1, 3, 32, 32])
    test_x = test_x.transpose([0, 2, 3, 1])

    return test_x, test_y