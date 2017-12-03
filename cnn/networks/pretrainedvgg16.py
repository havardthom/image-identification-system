#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VGG16 model for Keras. Reference:
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from keras.models import Sequential
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.data_utils import get_file

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'

class PretrainedVGG16NoTop:
	@staticmethod
	def build(depth, width, height):
		# Build the VGG16 network
		model = Sequential()
		model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(depth, width, height)))
		model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		# Load the weights of the VGG16 net (trained on ImageNet)
		weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
								TH_WEIGHTS_PATH_NO_TOP,
								cache_subdir='models')
		model.load_weights(weights_path)

		return model
