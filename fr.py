from __future__ import division
import os
import tensorflow as tf
import numpy as np
# Precessing images to get low-frequency part or high-frequency part

def split_image_concat(inputs):
    ###inputs [batch_size,h,w,c]
    R_img,G_img,B_img = tf.split(inputs,[1,1,1],3)
    ##reduce dim
    R_img = tf.squeeze(R_img,3)
    G_img = tf.squeeze(G_img,3)
    B_img = tf.squeeze(B_img,3)
    ##expand dim
    R_img = tf.expand_dims(R_img,1)
    G_img = tf.expand_dims(G_img,1)
    B_img = tf.expand_dims(B_img,1)
    ###[batch_size,c,h,w]
    ret_img = tf.concat([R_img,G_img,B_img],1)

    return ret_img

def recovery_images(inputs):
    ###inputs [batch_size,c,h,w]
    R_img,G_img,B_img = tf.split(inputs,[1,1,1],1)
    ##reduce dim
    R_img = tf.squeeze(R_img,1)
    G_img = tf.squeeze(G_img,1)
    B_img = tf.squeeze(B_img,1)
    ##expand dim
    R_img = tf.expand_dims(R_img,3)
    G_img = tf.expand_dims(G_img,3)
    B_img = tf.expand_dims(B_img,3)
    ###[batch_size,h,w,c]
    ret_img = tf.concat([R_img,G_img,B_img],3)

    return ret_img

def generate_positive_negative_mask(shape):
    ret = np.ones(shape,dtype = np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (i+j) %2 !=0:
                ret[i,j] = -1

    return ret

def diamond_binary_mask(shape,start,center):
    end = 2*center-start
    ret = np.zeros(shape,dtype = np.float32)
    count = 0
    area = 0
    for i in range(start,center+1):
        for j in range(center-count,center+count+1):
            ret[i,j]=1
            area =area+1
        count =count+1

    count =center-start-1
    for i in range(center+1,end+1):
        for j in range(center-count,center+count+1):
            ret[i][j]=1
            area =area+1
        count = count-1

    return ret,area

def center_mask(scale_size,shape=None): #scale_size=2n*1
    if shape == None:
        shape = [64, 64]
    mask = np.zeros(shape, dtype=np.float32)
    area = 0
    if scale_size > 0:
        center=shape[0]/2+1-1
        start_num = int(center-(scale_size-1)/2)
        end_num = int(center+(scale_size-1)/2 +1)
        for j in range(start_num,end_num):
            for i in range(start_num,end_num):
                mask[i][j]=1
                area= area+1
    elif scale_size ==-1:
        mask = np.ones(shape, dtype=np.float32)
        area = shape[0]*shape[1]
    else:
        pass
    return mask,area

def fft(images,scale_size1, scale_size2,shape=None):
    if shape==None:
        shape=[64,64]
    convert_images = split_image_concat(images)
    convert_images_shift = tf.multiply(generate_positive_negative_mask(shape), convert_images)

    fft_true = tf.fft2d(tf.cast(convert_images_shift, tf.complex64))
    mask1, area1 = center_mask(scale_size1,shape)
    mask2, area2 = center_mask(scale_size2,shape)
    # print(np.sum(mask2))
    mask2 = mask2 - mask1
    mask22 = tf.complex(mask2, tf.zeros_like(mask2))
    fft_true2 = tf.multiply(mask22, fft_true)
    ifft_image2 = recovery_images(tf.multiply(generate_positive_negative_mask(shape),
                                                      tf.cast(tf.ifft2d(fft_true2), tf.float32))
                                          )
    return ifft_image2

