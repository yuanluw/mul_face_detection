# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 15:19, matt '

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from pre_process import get_dataset
import model
import config
from utils import Display_board, FocalLoss, accuracy, count_time

cur_path = os.path.abspath(os.path.dirname(__file__))


def train(net, train_data, optimizer, criterion, arg):

    feature_net = net[0]
    metric_fc = net[1]
    feature_net = feature_net.cuda()
    metric_fc = metric_fc.cuda()
    best_state_dict = 0

    if arg.use_visdom:
        viz = Display_board(env_name="train", port=arg.port)
        train_acc_win = viz.add_Line_windows(name="train_pixel_acc")
        train_loss_win = viz.add_Line_windows(name="train_loss")
        train_y_axis = 0
    print("start training: ", datetime.now())

    for epoch in range(arg.epochs):
        # train stage
        train_loss = 0.0
        train_acc = 0.0
        feature_net = feature_net.train()
        i = 0
        if arg.use_visdom is not True:
            prev_time = datetime.now()
        for im, label in train_data:
            i += 1  # train number
            im = Variable(im.cuda())
            label = Variable(label.cuda())

            out = feature_net(im)
            out = metric_fc(out, label)

            loss = criterion(out, label)
            # if hasattr(torch.cuda, "empty_cache"):
            #   torch.cuda.empty_cache()
            loss.backward()
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(feature_net.parameters(), config.grad_clip)
            optimizer.step()

            cur_loss = loss.item()
            cur_acc = accuracy(out, label)
            train_loss += cur_loss
            train_acc += cur_acc

            if i % config.print_freq == 0:
                # visualize curve
                if arg.use_visdom:
                    train_y_axis += 1
                    viz.update_line(w=train_acc_win, Y=cur_acc, X=train_y_axis)
                    viz.update_line(w=train_loss_win, Y=cur_loss, X=train_y_axis)
                else:
                    now_time = datetime.now()
                    time_str = count_time(prev_time, now_time)
                    print("train: current (%d/%d) batch loss is %f pixel acc is %f time "
                          "is %s" % (i, len(train_data), train_loss/i, train_acc/i, time_str))
                    prev_time = now_time

        print("train: the (%d/%d) epochs acc: %f loss: %f, cur time: %s" % (epoch,
              arg.epochs, train_acc/i, train_loss/i, str(datetime.now())))

        torch.save(best_state_dict, os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl")))
        torch.save(best_state_dict, os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl")))

    print("end time: ", datetime.now())
    torch.save(best_state_dict, os.path.join(cur_path, "pre_train", str(arg.net + "end_.pkl")))
    torch.save(best_state_dict, os.path.join(cur_path, "pre_train", str("metric_fc" + "end_.pkl")))


def run(arg):
    print("lr %f, epoch_num %d, decay_rate %f pre_train %d gamma %f" %
          (arg.lr, arg.epochs, arg.decay, arg.pre_train, arg.gamma))

    train_data = get_dataset(arg, config.data_path, index="train")

    feature_net = model.get_model(arg.net)

    metric_fc = model.ArcMarginModel(emb_size=config.emb_size, easy_margin=config.easy_margin, margin_m=config.margin_m,
                                     margin_s=config.margin_s)
    if arg.mul_gpu:
        feature_net = nn.DataParallel(feature_net)
        metric_fc = nn.DataParallel(metric_fc)

    if arg.pre_train:
        feature_net.load_state_dict(torch.load(os.path.join(cur_path, "pre_train", str(arg.net + "_.pkl"))))
        metric_fc.load_state_dict(torch.load(os.path.join(cur_path, "pre_train", str("metric_fc" + "_.pkl"))))

    print('Total params: %.2fM' % (sum(p.numel() for p in feature_net.parameters()) / 1000000.0))
    optimizer = optim.Adam([{'params': feature_net.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=arg.lr, weight_decay=arg.decay)

    if arg.use_focal_loss:
        criterion = FocalLoss(gamma=arg.gamma).to(config.device)
    else:
        criterion = nn.CrossEntropyLoss().to(config.device)

    train((feature_net, metric_fc), train_data, optimizer, criterion, arg)


