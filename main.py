import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import frgan
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--scale_size', dest='scale_size', type=int, default=64, help='# images in batch')
parser.add_argument('--z_num', dest='z_num', type=int, default=64, help='# z num in dis')
parser.add_argument('--hidden_num', dest='hidden_num', type=int, default=128, help='# number of channel')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--low_rate', dest='low_rate', type=float, default=1.0, help='low frequency rate')
parser.add_argument('--high_rate', dest='high_rate', type=float, default=0.0, help='high hz rate')
parser.add_argument('--logs_dir', dest='logs_dir', default='./logs', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test results are save here')
parser.add_argument('--g_lr', dest='g_lr', type=float, default=0.00008, help='insavitial g learning rate for adam')
parser.add_argument('--d_lr', dest='d_lr', type=float, default=0.00008, help='initial d learning rate for adam')
parser.add_argument('--lr_lower_boundary', dest='lr_lower_boundary', type=float, default=0.00002, help='lower boundary of learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='momentum term of adam')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.5, help='rate')
parser.add_argument('--lambda_k', dest='lambda_k', type=float, default=0.001, help='the proportional gain for k')
parser.add_argument('--max_step', dest='max_step', type=int, default=6000000, help='max training steps')
parser.add_argument('--lr_update_step', dest='lr_update_step', type=int, default=100000, help='updating lr every default steps')
parser.add_argument('--log_step', dest='log_step', type=int, default=50, help='logs are saved every default steps')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    model = frgan(args)
    model.train() if args.phase == 'train' \
            else model.test()

if __name__ == '__main__':
    tf.app.run()

# CUDA_VISIBLE_DEVICES=2 python main.py
# tensorboard --port=6010 --logdir=./logs
