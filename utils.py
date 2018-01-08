"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
from scipy.io import loadmat as load
import numpy as np
import copy
from time import gmtime, strftime
from PIL import Image
import os
from glob import glob
import tensorflow as tf
# Processing images and loading images

def get_loader(batch_size,scale_size=64,seed=None):
    dir_path = '/home/guangyuan/CelebA/Img/img_align_celeba'
    names = os.listdir(dir_path)
    paths = []
    for name in names:
        src_path = dir_path + '/' + name
        paths.append(src_path)

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape=[h,w,3]

    filename_queue = tf.train.string_input_producer(list(paths),shuffle=False, seed = seed)
    reader =tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    tf_decode =tf.image.decode_jpeg
    image = tf_decode(data, channels= 3)
    image.set_shape(shape)
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image],batch_size=batch_size,num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue,name='synthetic_inputs'
    )
    queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
    queue =tf.image.resize_nearest_neighbor(queue,[scale_size, scale_size])
    return tf.to_float(queue)

def make_grid(tensor,nrow=8,padding=2):
    nmaps = tensor.shape[0]
    xmaps = min(nmaps,nrow)
    ymaps = int(math.ceil(float(nmaps)/xmaps))
    hp, wp = int(tensor.shape[1]+padding), int(tensor.shape[2]+padding)
    grid = np.zeros([ymaps*hp + 1 + padding//2, xmaps*wp + 1 + padding//2, 3],dtype=np.uint8)
    k=0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, w =y*hp + 1 + padding//2, x*wp + 1 + padding//2
            grid[h:h+tensor.shape[1],w:w+tensor.shape[2]] = tensor[k]
            k=k+1

    return grid

def save_image(tensor,filename, nrow=8, padding=2):

    ndarr = make_grid(tensor,nrow=nrow,padding=padding)

    im =Image.fromarray(ndarr)

    im.save(filename)

def norm_img(image):
    return image/127.5-1

def denorm_img(norm):
    return tf.clip_by_value((norm+1)*127.5,0,255)

def denorm_img_nc(norm):
    return 255*(norm-tf.reduce_min(norm))/(tf.reduce_max(norm)-tf.reduce_min(norm))

