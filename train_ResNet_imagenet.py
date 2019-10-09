from models.ResNet_imagenet import ResNet
from utils.load_imagenet import preprocess
from config.config_ResNet_imagenet import cfg
import time

if __name__ == '__main__':
    # cfg.batch_size = 32
    # cfg.train_num = 1281167
    # cfg.epochs = 100

    # tfrecords 파일로 변환
    net = ResNet('resnet101', True)

    for epoch in range(cfg.epochs):
        loss = 0
        acc = 0
        for step in range(int(cfg.train_num / cfg.batch_size)):
            start_time = time.time()
            if step % 200 == 0 and step != 0:
                global_step, train_loss, train_acc, lr = net.train(True)
            else:
                global_step, train_loss, train_acc, lr = net.train(False)
            end_time = time.time()
            print('Epoch {} step {}, loss = {}, acc = {} , processing time = {} lr = {}'.format(epoch, global_step, train_loss, train_acc, end_time- start_time, lr))
            loss += train_loss
            acc += train_acc

        print('Epoch {} step {}, loss = {}, acc = {}'.format(epoch, global_step, loss / step, acc / step))
