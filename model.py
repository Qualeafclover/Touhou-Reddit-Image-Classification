from configs import *
import os
import numpy as np
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Lambda, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import ReLU, Softmax, BatchNormalization, Dropout, Input
from tensorflow.keras import Model
import tensorflow_addons as tfa



def process_img(input_data):
    return tf.image.resize(input_data, (500, 500))

def convolutional(input_data, filters:int, kernel_size=3, dropout=0.0,
                  downsample=False, bn=True, mish=True, name='convolutional'):
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    output_data = Conv2D(filters=filters, kernel_size=kernel_size, name=name,
                         strides=strides, padding='SAME', use_bias=not bn)(input_data)
    if bn:
        output_data = BatchNormalization()(output_data)
    if mish:
        output_data = tfa.activations.mish(output_data)
    if dropout != 0.0:
        output_data = Dropout(dropout)(output_data)
    return output_data

def main_model(input_data):
    output_data = process_img(input_data)

    output_data = convolutional(output_data, filters=64, kernel_size=7, downsample=True)
    print(output_data.shape)

    return output_data

def create_model():
    input_layer = Input([3000, 3000, 3])
    output_tensors = main_model(input_layer)
    return tf.keras.Model(input_layer, output_tensors)

if __name__ == '__main__':
    model = create_model()

    from pixiv_grabber import PixivGrabber
    grabber = PixivGrabber(init_driver=False)
    grabber.load_info()

    for dataset in grabber:
        X, y = dataset
        print(X.shape)
        print(y.shape)

