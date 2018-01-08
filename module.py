from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
# Building discriminator and generator
# Defining kinds of loss criterions

def discriminator(x, z_num,hidden_num,repeat_num,name="D",reuse= False):

    with tf.variable_scope(name,reuse=reuse) as vs:
        x = slim.conv2d(x,hidden_num,3,1,activation_fn=tf.nn.elu)
        for idx in range(repeat_num):
            channel_num = hidden_num*(idx + 1)
            x = slim.conv2d(x,channel_num,3,1,activation_fn=tf.nn.elu)
            x = slim.conv2d(x,channel_num,3,1,activation_fn=tf.nn.elu)
            if idx < repeat_num-1:
                x = slim.conv2d(x,channel_num,3,2,activation_fn=tf.nn.elu)
        x= tf.reshape(x,[-1,np.prod([8,8,channel_num])])
        z= x = slim.fully_connected(x,z_num,activation_fn=None)

        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = tf.reshape(x, [-1, 8, 8, hidden_num])
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num - 1:
                x = upscale(x, 2)
        out = slim.conv2d(x, 3, 3, 1, activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    return out,z, variables

def generator(z,hidden_num,repeat_num, reuse=False):

    with tf.variable_scope("G",reuse=reuse) as vs:
        num_output = int(np.prod([8,8,hidden_num]))
        x = slim.fully_connected(z,num_output, activation_fn=None)
        x = tf.reshape(x,[-1,8,8,hidden_num])
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num -1:
                x = upscale(x,2)
        out = slim.conv2d(x, 3,3,1,activation_fn=None)
    variables = tf.contrib.framework.get_variables(vs)
    # print(out.get_shape())
    return out, variables
def upscale(x,scale):
    _, h, w, _ = int_shape(x)
    return tf.image.resize_nearest_neighbor(x,(h*scale,w*scale))
def int_shape(x):
    shape= x.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
