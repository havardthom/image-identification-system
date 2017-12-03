#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Custom model"""
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout

class CustomModel:
	@staticmethod
	def build(depth, width, height, nb_classes):
		# Build the custom model
		model = Sequential()
		model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(depth, height, width)))
		model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
		model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0.5))

		if nb_classes == 2:
			model.add(Dense(1, activation='sigmoid'))
		else:
			model.add(Dense(nb_classes, activation='softmax'))

		return model
