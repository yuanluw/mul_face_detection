# -*- encoding: utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/4/9 18:17, matt '


import argparse


def get_augments():
    parser = argparse.ArgumentParser(description="pytorch mul face classification")

    parser.add_argument("--action", type=str, default="train", choices=("train",))
    parser.add_argument("--mul_gpu", type=int, default=0, help="use multiple gpu(default: 1")
    parser.add_argument("--net", type=str, default="MNet", choices=("resnet50", "resnet101"))

    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate(default: 0.1)")
    parser.add_argument("-lr_step", type=int, default=10, help="period of learning rate decay")
    parser.add_argument("--optimizer", default="adam", choices=("sgd", "adam"), help='optimizer')
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')
    parser.add_argument('--use_focal_loss', type=bool, default=False, help='focal loss')
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")

    parser.add_argument("--use_visdom", type=int, default=1)
    parser.add_argument("--port", type=int, default=8097)

    return parser.parse_args()


def main():
    arg = get_augments()
    if arg.action == "train":
        from train import run
        run(arg)


if __name__ == "__main__":
    main()

