# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 15:19, matt '

import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from pre_process import get_dataset
import model
import config
from utils import Display_board, FocalLoss, accuracy, count_time, get_logger, AverageMeter, get_visdom, save_checkpoint

cur_path = os.path.abspath(os.path.dirname(__file__))


def run(arg):
    torch.manual_seed(7)
    np.random.seed(7)
    print("lr %f, epoch_num %d, decay_rate %f gamma %f" % (arg.lr, arg.epochs, arg.decay, arg.gamma))

    start_epoch = 0

    train_data = get_dataset(arg, config.data_path, index="train",
                             input_shape=(config.channel, config.image_h, config.image_w))

    if arg.checkpoint is None:
        feature_net = model.get_model(arg.net)
        metric_fc = model.ArcMarginModel(emb_size=config.emb_size, easy_margin=config.easy_margin, margin_m=config.margin_m,
                                         margin_s=config.margin_s)
        if arg.mul_gpu:
            feature_net = nn.DataParallel(feature_net)
            metric_fc = nn.DataParallel(metric_fc)

        if arg.optimizer == 'sgd':
            optimizer = optim.SGD([{'params': feature_net.parameters()}, {'params': metric_fc.parameters()}],
                                  lr=arg.lr, momentum=arg.momentum, weight_decay=arg.decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=arg.lr_step, gamma=0.1)
        elif arg.optimizer == "adam":
            optimizer = optim.Adam([{'params': feature_net.parameters()}, {'params': metric_fc.parameters()}],
                                      lr=arg.lr, weight_decay=arg.decay)

    else:
        checkpoint = torch.load(os.path.join(config.checkpoint_path, arg.checkpoint))
        start_epoch = checkpoint['epoch'] + 1
        feature_net = checkpoint['feature_net']
        metric_fc = checkpoint['metric_fc']
        optimizer = checkpoint['optimizer']

    feature_net = feature_net.to(config.device)
    metric_fc = metric_fc.to(config.device)

    logger = get_logger()
    if arg.use_visdom:
        vis = get_visdom(port=arg.port, env_name="mul_face_detection_train")
    else:
        vis = None
    if arg.use_focal_loss:
        criterion = FocalLoss(gamma=arg.gamma).to(config.device)
    else:
        criterion = nn.CrossEntropyLoss().to(config.device)

    print('Total params: %.2fM' % (sum(p.numel() for p in feature_net.parameters()) / 1000000.0))
    print("start training: ", datetime.now())
    for epoch in range(start_epoch, arg.epochs):
        prev_time = datetime.now()
        train_loss, train_acc = train(train_data, feature_net, metric_fc, criterion, optimizer, epoch, logger, vis)
        now_time = datetime.now()
        time_str = count_time(prev_time, now_time)
        print("train: current (%d/%d) batch loss is %f pixel acc is %f time "
              "is %s" % (epoch, arg.epochs, train_loss, train_acc, time_str))
        if arg.optimizer == "sgd":
            scheduler.step()
        if epoch % 10 == 0:
            save_checkpoint(epoch, feature_net, metric_fc, optimizer, train_acc, False)


def train(train_data, feature_net, metric_fc, criterion, optimizer, epoch, logger, vis):
    feature_net.train()
    metric_fc.train()
    losses = AverageMeter()
    top1_accs = AverageMeter()

    for i, (img, label) in enumerate(train_data):
        img = img.to(config.device)
        label = label.to(config.device)

        feature = feature_net(img)
        output = metric_fc(feature, label)

        # calculate loss
        loss = criterion(output, label)

        # bp
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

        # metrics track
        losses.update(loss.item())
        top1_accuracy = accuracy(output, label, 1)
        top1_accs.update(top1_accuracy)

        if i % config.print_freq == 0:
            logger.info('Epoch: [{0}][{1}][{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Top1 accuracy {top1_accs.val:.3f} ({top1_accs.avg:.3f})'
                        .format(epoch, i, len(train_data), loss=losses, top1_accs=top1_accs))
            if vis is not None:
                vis.update(losses.avg, top1_accs.avg)

    return losses.avg, top1_accs.avg



